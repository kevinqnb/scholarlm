from pydantic import BaseModel
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from .contextlm2 import ContextLM2


def response_validator(response_structure, response):
    pyd = response_structure.model_validate_json(response)
    out_dict = pyd.model_dump()
    return out_dict

class BooleanResponse(BaseModel):
    answer: bool

class DataPointResponse(BaseModel):
    value: float | str | None
    units: str | None



class MeasurementLM:
    """
    A language model class designed for organized collection of measurements from scientific text.

    Args:
        model_name (str): The name or path of the pre-trained language model from the huggingface 
            collection.
        item_description (str): Main description for the items to be measured.
        identification_schema (dict[str, str]): A dictionary defining the identification schema, 
            where keys are the measurement identifiers and values are their descriptions.
        measurement_schema (dict[str, str]): A dictionary defining the measurement schema, 
            where keys are

    Attributes:

    """
    def __init__(
        self,
        model_name: str,
        item_description: str,
        identification_schema: dict[str, str],
        measurement_schema: dict[str, str],
        sampling_params: dict[str, any] = None,
        return_full_output: bool = False,
    ):
        self.model_name = model_name
        self.item_description = item_description
        self.identification_schema = identification_schema
        self.measurement_schema = measurement_schema
        self.sampling_params = {
            "temperature" : 0.90,
            "top_p" : 0.95,
            "top_k" : 64,
            "repetition_penalty" : 1.0,
            "max_tokens" : 2048,
        } | sampling_params
        self.return_full_output = return_full_output

        self.llm = LLM(model=model_name)

    
    def _filter(self):
        """
        Filters the input text chunks to retain only those relevant to the item of interest.

        Args:
            
        Returns:
            
        """        
        messages = []
        for i, datapoint in enumerate(self.data):
            instructions = (
                f"You are an expert at identifying relevant information in scientific texts. "
                f"Determine if the given context contains any relevant information. "
                f"Respond 'true' if the context is relevant and 'false' if it is not. "
            )
            context = datapoint['context']
            query = "Is the context relevant to measuring or identifying " + f"{', '.join(self.item_description)}?"
            prompt = (
                f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
            )
            messages.append([
                {"role": "user", "content": prompt}]
            )

        guided_decoding_params = GuidedDecodingParams(
            choice = ['true', 'false']
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )

        responses = self.llm.chat(messages = messages, sampling_params = sampling_params)
        response_texts = [r.outputs[0].text for r in responses]

        filtered_data = []
        for i, resp in enumerate(response_texts):
            if resp == 'true':
                filtered_data.append(self.data[i])
        
        return filtered_data
    

    def _identify(self):
        """
        Identifies items in the text chunks based on the identification schema.

        Args:
            
        Returns:
            
        """
        class IdentificationList(BaseModel):
            items: list[self.identification_schema]

        #identification_schema_json = self.identification_schema.model_json_schema()
        identification_list_json = IdentificationList.model_json_schema()
        identification_prompt = self.identification_schema.model_config['prompt']
        messages = []
        for i, datapoint in enumerate(self.data):
            instructions = identification_prompt
            context = datapoint['context']
            query = "Follow the instructions to identify the items mentioned in the context: "
            prompt = (
                f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
            )
            messages.append([
                {"role": "user","content": prompt}]
            )

        guided_decoding_params = GuidedDecodingParams(
            json=identification_list_json
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )

        responses = self.llm.chat(messages = messages, sampling_params = sampling_params)
        response_texts = [r.outputs[0].text for r in responses]
        response_validated = []
        for r in response_texts:
            try:
                resp_validated = response_validator(IdentificationList, r)
            except:
                print("Validation error in identification response.")
                resp_validated = {'items': []}
            response_validated.append(resp_validated)

        # Collect all items across each paper
        paper_items = {}
        for i, resp in enumerate(response_validated):
            datapoint = self.data[i]
            document_id = datapoint['document_id']
            for item in resp['items']:
                #itemized_data.append(datapoint | item)
                if document_id not in paper_items:
                    paper_items[document_id] = []
                
                if not any(item == added_item for added_item in paper_items[document_id]):
                    paper_items[document_id].append(item)


        # De-duplicate items within each paper
        for doc_id, doc_items in paper_items.items():
            unique_items = self._deduplicate(doc_items)
            paper_items[doc_id] = unique_items

        itemized_data = []
        for i, datapoint in enumerate(self.data):
            document_id = datapoint['document_id']
            for item in paper_items.get(document_id, []):
                itemized_data.append(datapoint | item)

        # De-duplicate itemized data points
        #unique_itemized_data = [dict(s) for s in {frozenset(d.items()) for d in itemized_data}]

        #return unique_itemized_data
        return itemized_data
    

    def _deduplicate(self, items: list[dict]):
        """
        De-duplicates a list of dictionaries.

        Args:
            items (list[dict]): A list of dictionaries to be de-duplicated.
        Returns:
            unique_items (list[dict]): A de-duplicated list of dictionaries.
        """
        matches = {}
        for i, item in enumerate(items):
            item_name = item.get('name', None)
            if item_name is not None and item_name is not 'None':
                if item_name not in matches:
                    matches[item_name] = [item]
                else:
                    matches[item_name].append(item)

        unique_items = []
        for name, name_items in matches.items():
            uitem = {}
            for feature in self.identification_schema.model_fields.keys():
                feature_values = [ni.get(feature, None) for ni in name_items if ni.get(feature, None) is not None and ni.get(feature, None) != 'None']
                if len(feature_values) > 0:
                    consensus_value = max(set(feature_values), key=feature_values.count)
                    uitem[feature] = consensus_value
                else:
                    uitem[feature] = 'None'
            unique_items.append(uitem)

        return unique_items


    
    '''
    def _deduplicate(self, items: list[dict]):
        """
        De-duplicates a list of dictionaries.

        Args:
            items (list[dict]): A list of dictionaries to be de-duplicated.
        Returns:
            unique_items (list[dict]): A de-duplicated list of dictionaries.
        """
        class IdentificationList(BaseModel):
            items: list[self.identification_schema]

        instructions = (
            f"You are an expert in data processing and collection. "
            f"Given a list of entities described by a common set of attributes, your task is to "
            f"identify the set of unique entities by consolidating and removing any duplicates. "
            f"Two entities are considered exact duplicates if they share identical values for all attributes. "
            f"Duplicates, however, may also have minor variations in certain attributes due to differences in formatting, "
            f"abbreviations, or reporting errors. Likewise, some entities may be missing values for certain "
            f"attributes, but can still be considered duplicates if the available information matches closely enough. "
            f"To determine if two entities are duplicates, consider the overall similarity of their attributes, "
            f"and use your best judgement by taking into account potential variations and missing data. "
            f"For a set of duplicate entities, retain only a single representative entity in the final output. "
            f"This representative entity should ideally contain the most complete and accurate information available among the duplicates. "
            f"Do NOT add any new attributes or information that is not already present in the provided entities. "
            f"Do NOT infer or fabricate any values. "
            f"Do NOT remove entities which are not duplicates. "
            f"You will be provided with a list of entities in JSON format, and your task is to return the list of unique entities with duplicates removed "
            f"in the same format. "
        )
        context = f"The list of entities is as follows: {items}"
        print(context)
        print()
        query = "Identify and return the list of unique entities from the provided list."
        prompt = (
            f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
        )
        message = [
            {"role": "user","content": prompt}
        ]
        #identification_schema_json = self.identification_schema.model_json_schema()
        identification_list_json = IdentificationList.model_json_schema()
        guided_decoding_params = GuidedDecodingParams(
            #json=identification_schema_json
            json=identification_list_json
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )

        responses = self.llm.chat(messages = message, sampling_params = sampling_params)
        response_text = responses[0].outputs[0].text
        try:
            #resp_validated = response_validator(self.identification_schema, response_text)
            resp_validated = response_validator(IdentificationList, response_text)
        except:
            print("Validation error in identification response, reverting to original list.")
            resp_validated = {'items': items}

        unique_items = resp_validated['items']
        print(f"Reduced {len(items)} items to {len(unique_items)} unique items.")
        print(unique_items)
        print()
        print()
        return unique_items
    '''
    

    def _measurements_filter(self):
        """
        Filters the input items to retain only those relevant for measurements.

        Args:
            
        Returns:
            
        """
        messages = []
        message_measurement_types = []
        message_data_ids = []
        for m in self.measurement_schema.model_fields.keys():
            m_description = self.measurement_schema.model_fields[m].description
            for i, datapoint in enumerate(self.data):
                item = {k: v for k,v in datapoint.items() if k not in ['context', 'chunk_id', 'document_id']}
                instructions = (
                    f"You are an expert in discerning whether or not a given piece of scientific text is relevant for data collection. "
                    f"You will be given context from a research paper, along with a description of a feature to be measured for a specific entity. "
                    f"Your task is to determine if the context contains a numerical measurement for the given feature and entity. "
                    f"Respond 'false' if the the given entity or measurement feature do not appear in the context. "
                    f"Respond 'false' if the context does not explicity provide data for the given feature and entity. "
                    f"Respond 'false' if the data reported is not either a direct numerical measurement or a mean of numerical measurements. "
                    f"Respond 'false' for if the data reported only contains values for parameter estimates or other statistical measures of fit. "
                    f"Respond 'false' for ranges of values, inequalties, or other cases where there is not a clear choice for a single numerical value. "
                    f"Respond 'true' only if the context explicity provides a direct numerical value measured for the given feature, with respect to the entity in question. "
                )
                context = datapoint['context']
                query = "Does the context contain data for " + f"{m_description}  the entity {item}?"
                prompt = (
                f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
                )
                messages.append([
                    {"role": "user","content": prompt}]
                )
                message_measurement_types.append(m)
                message_data_ids.append(i)

        guided_decoding_params = GuidedDecodingParams(
            choice = ['true', 'false']
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )

        responses = self.llm.chat(messages = messages, sampling_params = sampling_params)
        response_texts = [r.outputs[0].text for r in responses]

        measurement_data = []
        for i, resp in enumerate(response_texts):
            if resp == 'true':
                measurement_data.append(
                    self.data[message_data_ids[i]] | {'measurement_id': i, 'measurement': message_measurement_types[i]}
                )

        return measurement_data
    

    def _measure(self):
        """
        Extracts measurements from the text chunks for the identified items.

        Args:

        Returns:
            
        """
        instructions = (
            f"You are an expert in extracting precise numerical data from user provided, scientific text. "
            f"You will be queried with a description of an specific entity to be measured, along with the measurement type to report for. "
            f"Your task is to extract the corresponding value if it appears in the provided context. "
            f"Respond 'None' if the the given entity or measurement feature do not appear in the context. "
            f"Respond 'None' if the context does not explicity provide data for the given feature and entity. "
            f"Respond 'None' if the data reported is not either a direct numerical measurement or a mean of numerical measurements. "
            f"Respond 'None' for if the data reported only contains values for parameter estimates or other statistical measures of fit. "
            f"Respond 'None' for ranges of values, inequalties, or other cases where there is not a clear choice for a single numerical value. "
            f"Respond with the extracted value only if the context explicity provides a direct numerical value measured for the given feature, with respect to the entity in question. "
            f"Copy the value exactly as it appears in the context. "
            f"Give the value only, and do not include any units of measurement, descriptors, or explanation in your response. "
            f"If the value is associated with uncertainty measures (e.g., ± values, confidence intervals), report only the central value without any uncertainty information. "
        )
        messages = []
        for i, datapoint in enumerate(self.data):
            item = {k: v for k,v in datapoint.items() if k not in ['context', 'chunk_id', 'document_id', 'measurement', 'measurement_id']}
            measurement = datapoint['measurement']
            context = datapoint['context']
            query = "Extract the value of " + f"{measurement} for the entity {item}."
            messages.append((instructions, context, query))

        ctxlm_params = {k: v for k,v in self.sampling_params.items() if k not in ['max_tokens', 'seed', 'temperature', 'stop']}
        ctxlm_params['do_sample'] = False
        ctxlm_params['max_new_tokens'] = 20
        ctxlm = ContextLM2(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            top_k = 10,
            sampling_params=ctxlm_params,
            return_full_output=False,
            verbose = False,
            cache_output_dir="data/pond_adversarial_test" # REMEMBER THAT THIS IS HARDCODED RIGHT NOW
        )
        measurement_responses = ctxlm.predict(messages)

        measured_data = []
        '''
        for i,response_dict in enumerate(measurement_responses):
            if response_dict['response'].strip().lower() != 'none':
                measured_data.append(
                    self.data[i] | 
                    {
                        'value': response_dict['response'],
                        'context_scores' : response_dict.get('context_scores', {}),
                        'parametric_scores' : response_dict.get('parametric_scores', {}),
                        'copying_scores' : response_dict.get('copying_scores', {}),
                        'linear_probes' : response_dict.get('linear_probes', {})
                    }
                )
        '''
        for i, response in enumerate(measurement_responses):
            if response.strip().lower() != 'none':
                measured_data.append(
                    self.data[i] | 
                    {
                        'value': response,
                    }
                )

        return measured_data
    

    def _measure_vllm(self):
        """
        Extracts measurements from the text chunks for the identified items.

        Args:

        Returns:
            
        """
        instructions = (
            f"You are an expert in extracting precise numerical data from user provided, scientific text. "
            f"You will be queried with a description of an specific entity to be measured, along with the measurement type to report for. "
            f"Your task is to extract the corresponding value if it appears in the provided context. "
            f"Respond 'None' if the the given entity or measurement feature do not appear in the context. "
            f"Respond 'None' if the context does not explicity provide data for the given feature and entity. "
            f"Respond 'None' if the data reported is not either a direct numerical measurement or a mean of numerical measurements. "
            f"Respond 'None' for if the data reported only contains values for parameter estimates or other statistical measures of fit. "
            f"Respond 'None' for ranges of values, inequalties, or other cases where there is not a clear choice for a single numerical value. "
            f"Respond with the extracted value only if the context explicity provides a direct numerical value measured for the given feature, with respect to the entity in question. "
            f"Copy the value exactly as it appears in the context. "
            f"Give the value only, and do not include any units of measurement, descriptors, or explanation in your response. "
        )
        messages = []
        for i, datapoint in enumerate(self.data):
            item = {k: v for k,v in datapoint.items() if k not in ['context', 'chunk_id','document_id', 'measurement']}
            measurement = datapoint['measurement']
            context = datapoint['context']
            query = "Extract the value of " + f"{measurement} for the entity {item}."
            prompt = (
                f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
                )
            messages.append([
                {"role": "user","content": prompt}]
            )

        mlm_params = {k: v for k,v in self.sampling_params.items() if k not in ['max_tokens', 'seed', 'temperature', 'stop']}
        mlm_params['temperature'] = 0.0
        mlm_params['max_tokens'] = 20
        sampling_params = SamplingParams(
            **mlm_params
        )

        responses = self.llm.chat(messages = messages, sampling_params = sampling_params)
        response_texts = [r.outputs[0].text for r in responses]
        measured_data = []
        for i, resp in enumerate(response_texts):
            if resp.strip().lower() != 'none':
                measured_data.append(
                    self.data[i] | {'value': resp}
                )

        return measured_data
    

    def _standardize(self):
        """
        Gives standardized units to the extracted measurements.

        Args:

        Returns:
            
        """
        instructions = (
            f"You are an expert in data collection and scientific measurements. "
            f"You will be given context from a research paper, along with a description of a measurement value and the feature and entity it was reported for. "
            f"Your task is to determine the unit of measurement for that data point by referencing the context, and then choosing from a list of given unit options. "
            f"To ensure units follow standard formatting conventions, your response should be limited to options from among the given list. "
            f"If, however, none of the options fit with what is seen in the context, respond with the unit exactly as it appears in the context. "
            f"Your response should include the unit only, do not include any additional explanation or text. "
        )        
        messages = []
        message_data_ids = []
        sampling_params = []
        for i, datapoint in enumerate(self.data):
            item = {k: v for k,v in datapoint.items() if k not in ['context', 'chunk_id', 'document_id', 'measurement', 'value', 'context_scores', 'parametric_scores', 'copying_scores', 'linear_probes']}
            measurement = datapoint['measurement']
            measurement_val = datapoint['value']

            measurement_description = self.measurement_schema.model_fields[measurement].description
            available_units = self.measurement_schema.model_fields[measurement].json_schema_extra.get('units', None)

            if available_units is not None:
                units_list = available_units #+ ['other']
                context = datapoint['context']
                query = (
                    f"Entity measured: {item}\n"
                    f"Measurement type: {measurement_description}\n"
                    f"Measurement value: {measurement_val}\n"
                    f"Determine the unit of measurement for the given data point from among the following choices: {available_units}."
                )
                prompt = (
                    f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
                )
                messages.append([
                    {"role": "user", "content": prompt}]
                )
                message_data_ids.append(i)
                #guided_decoding_params = GuidedDecodingParams(
                #    choice = units_list
                #)
                params = SamplingParams(
                    **self.sampling_params,
                    #guided_decoding=guided_decoding_params
                )
                sampling_params.append(params)

        responses = self.llm.chat(messages = messages, sampling_params = sampling_params)
        response_units = [r.outputs[0].text for r in responses]
        
        standardized_data = [datapoint for datapoint in self.data]
        for i, resp in enumerate(response_units):
            standardized_data[message_data_ids[i]]['units'] = resp.strip()

        return standardized_data
    

    def _judge(self):
        """
        Filters the input items to retain only those relevant for measurements.

        Args:
            
        Returns:
            
        """
        messages = []
        message_measurement_types = []
        message_data_ids = []

        for i, datapoint in enumerate(self.data):
            item = {k: v for k,v in datapoint.items() if k not in ['context', 'chunk_id', 'document_id', 'context_scores', 'parametric_scores', 'copying_scores', 'linear_probes']}
            instructions = (
                f"You are an expert in discerning textual accuracy for a data point extracted by an large language model. "
                f"You will be given context from a research paper, and asked to classify its relation to the extracted data point using the following categories:\n"
                f"hallucination: The extracted data point's 'value' feature does not explicity appear within the context.\n"
                f"disorientation: The data point appears to be derived from the context, but is incorrectly attributed to the given entity or measurement type.\n"
                f"deviation: The data point is generally supported by the context, but the given value is an aggregate statistic, range of values, inequality, or non-numerical description rather than a direct measurement.\n"
                f"valid: The data point is a direct measurement which is explicity supported by the context, and is made with respect to the correct entity and measurement type.\n\n"
                f"Respond by choosing the category which best describes the data point's relation to the given context."
            )
            context = datapoint['context']
            query = f"Given the context, which category best describes the following data point?: {item}?"
            prompt = (
            f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
            )
            messages.append([
                {"role": "user","content": prompt}]
            )

        guided_decoding_params = GuidedDecodingParams(
            choice = ['hallucination', 'disorientation', 'deviation', 'valid']
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )

        responses = self.llm.chat(messages = messages, sampling_params = sampling_params)
        response_texts = [r.outputs[0].text for r in responses]

        judged_data = [datapoint for datapoint in self.data]
        for i, resp in enumerate(response_texts):
            judged_data[i]['judgement'] = resp

        return judged_data


    def fit(
        self,
        chunks : list[list[str]],
    ):
        """
        Fits the MeasurementLM to the provided text chunks by filtering, identifying items, 
        and extracting measurements.

        Args:
            chunks (list[str]): A list of text chunks.
        Returns:
            measurements (list[dict]): A list of measurements extracted for identified items.
        """
        self.data = []
        for i in range(len(chunks)):
            #self.data.append({'chunk_id': i, 'context' : chunks[i]})
            for j, chunk in chunks[i].items():
                self.data.append({'document_id': i, 'chunk_id': j, 'context' : chunk})

        self.data = self._filter()
        self.data = self._identify()
        self.data = self._measurements_filter()
        if self.return_full_output:
            self.data = self._measure()
        else:
            self.data = self._measure_vllm()
        self.data = self._standardize()
        #self.data = self._judge()

        return self.data
    

    def save(self, filepath: str):
        """
        Saves the measurement data to a csv.

        Args:
            filepath (str): The path to the file where the data will be saved.
        """
        df = pd.DataFrame(self.data)
        df.to_csv(filepath, index=False)



