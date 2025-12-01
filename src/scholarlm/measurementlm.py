from pydantic import BaseModel
import pandas as pd
import torch
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
        sampling_params (dict[str, any]): A dictionary of sampling parameters for text generation.
        identification_prompt (str): A string containing the prompt instructions for item identification.
        identification_schema (BaseModel): A Pydantic BaseModel defining the identification schema. 
        measurement_schema (BaseModel): A Pydantic BaseModel defining the measurement schema.

    Attributes:

    """
    def __init__(
        self,
        model_name: str,
        identification_prompt: str,
        identification_schema: BaseModel,
        measurement_schema: BaseModel,
        sampling_params: dict[str, any] = {},
        return_full_output: bool = False,
        cache_dir: str | None = None,
    ):
        self.model_name = model_name
        self.sampling_params = {
            "temperature" : 0.90,
            "top_p" : 0.95,
            "top_k" : 64,
            "repetition_penalty" : 1.0,
            "max_tokens" : 2048,
        } | sampling_params
        self.identification_prompt = identification_prompt
        self.identification_schema = identification_schema
        self.entity_description = identification_schema.model_config['entity_description']
        self.primary_identifier = identification_schema.model_config['primary_identifier']

        assert self.primary_identifier in identification_schema.model_fields.keys(), "Primary identifier must be a valid field in the identification schema."

        self.measurement_schema = measurement_schema
        self.return_full_output = return_full_output
        self.cache_dir = cache_dir

        self.llm = LLM(model=model_name)

        '''
        ctxlm_params = {k: v for k,v in sampling_params.items() if k not in ['max_tokens', 'seed', 'temperature', 'stop']}
        ctxlm_params['do_sample'] = False
        ctxlm_params['max_new_tokens'] = 20
        self.ctxlm = ContextLM2(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            top_k = 20,
            sampling_params=ctxlm_params,
            nnsight_kwargs = {"torch_dtype": torch.bfloat16},
            cache_dir = self.cache_dir
        )
        '''

    
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
            query = "Is the context relevant to measuring or identifying " + f"{', '.join(self.entity_description)}?"
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
        messages = []
        for i, datapoint in enumerate(self.data):
            instructions = self.identification_prompt
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
            item_id = item.get(self.primary_identifier, None)
            if item_id is not None and item_id is not 'None':
                if item_id not in matches:
                    matches[item_id] = [item]
                else:
                    matches[item_id].append(item)

        unique_items = []
        for id, id_items in matches.items():
            uitem = {}
            for feature in self.identification_schema.model_fields.keys():
                feature_values = [x.get(feature, None) for x in id_items if x.get(feature, None) is not None and x.get(feature, None) != 'None']
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
                item = {k: v for k,v in datapoint.items() if k in self.identification_schema.model_fields.keys()}
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
                query = f"Does the context contain data for {m_description} measured for the entity {item}?"
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
        message_ids = []
        for i, datapoint in enumerate(self.data):
            item = {k: v for k,v in datapoint.items() if k in self.identification_schema.model_fields.keys()}
            measurement = datapoint['measurement']
            context = datapoint['context']
            query = "Extract the value of " + f"{measurement} for the entity {item}."
            messages.append((instructions, context, query))
            message_ids.append(datapoint['measurement_id'])

        measurement_responses = self.ctxlm.predict(messages, message_ids)

        measured_data = []
        
        for i, resp in enumerate(measurement_responses):
            if resp.strip().lower() != 'none':
                measured_data.append(
                    self.data[i] | 
                    {
                        'value': resp
                    }
                )
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
            item = {k: v for k,v in datapoint.items() if k in self.identification_schema.model_fields.keys()}
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
            item = {k: v for k,v in datapoint.items() if k in self.identification_schema.model_fields.keys()}
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
        '''
        self.data = []
        for i in range(len(chunks)):
            #self.data.append({'chunk_id': i, 'context' : chunks[i]})
            for j, chunk in chunks[i].items():
                self.data.append({'document_id': i, 'chunk_id': j, 'context' : chunk})
        '''
        self.data = []
        for i in range(len(chunks)):
            for j, page_chunks in chunks[i].items():
                for k, chunk in enumerate(page_chunks):
                    self.data.append({'document_id': i, 'page_id': j, 'chunk_id': k, 'context' : chunk})

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



