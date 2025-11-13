from pydantic import BaseModel
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from .contextlm import ContextLM


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
        identification_schema_json = self.identification_schema.model_json_schema()
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
            json=identification_schema_json
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
                resp_validated = response_validator(self.identification_schema, r)
            except:
                print("Validation error in identification response, assigning empty items list.")
                resp_validated = {'items': []}
            response_validated.append(resp_validated)

        itemized_data = []
        for i, resp in enumerate(response_validated):
            # Copy items found from each chunk across the rest of their corresponding paper
            #paper_id = self.data[i]['paper_id']
            #paper_data = [d for d in self.data if d['paper_id'] == paper_id]
            #for datapoint in paper_data:
            datapoint = self.data[i]
            for item in resp['items']:
                itemized_data.append(datapoint | item)

        # De-duplicate itemized data points
        unique_itemized_data = [dict(s) for s in {frozenset(d.items()) for d in itemized_data}]

        return unique_itemized_data
    

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
                item = {k: v for k,v in datapoint.items() if k not in ['context', 'chunk_id', 'paper_id']}
                instructions = (
                    f"You are an expert in discerning whether or not a given piece of scientific text is relevant for data collection. "
                    f"You will be given context from a research paper, along with a description of a feature to be measured for a specific entity. "
                    f"Your task is to determine if the context contains a numerical measurement for the given feature and entity. "
                    f"Respond 'false' if the the given entity or feature do not appear in the context. "
                    f"Respond 'false' if the context does not explicity provide data for the given feature and entity, or if it only reports aggregate statistics, a range of values, or an inequality. "
                    f"Respond 'true' only if the context explicity provides a distinct numerical measurement for the given feature, with respect to the entity in question. "
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
                    self.data[message_data_ids[i]] | {'measurement': message_measurement_types[i]}
                )

        return measurement_data
    

    def _measure(self):
        """
        Extracts measurements from the text chunks for the identified items.

        Args:

        Returns:
            
        """
        messages = []
        for i, datapoint in enumerate(self.data):
            item = {k: v for k,v in datapoint.items() if k not in ['context', 'chunk_id', 'measurement', 'paper_id']}
            measurement = datapoint['measurement']
            instructions = (
                f"You are an expert in extracting precise numerical data from user provided, scientific text. "
                f"A value is a single numerical measurement explicitly mentioned in the context. "
                f"You will be queried with a description of an specific entity to be measured, along with the measurement type to report for. "
                f"Your task is to extract the corresponding value from the provided context. "
                f"Copy the value exactly as it appears in the context. "
                f"Give the value only, and do not include any units of measurement, descriptors, or explanation in your response. "
                f"Respond 'None' if the requested information is not explicitly available in the given context."
            )
            context = datapoint['context']
            query = "Extract the value of " + f"{measurement} for the entity {item}."
            messages.append((instructions, context, query))

        ctxlm_params = {k: v for k,v in self.sampling_params.items() if k not in ['max_tokens', 'seed', 'temperature', 'stop']}
        ctxlm_params['do_sample'] = False
        ctxlm_params['max_new_tokens'] = 20
        ctxlm = ContextLM(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            top_k = 10,
            sampling_params=ctxlm_params,
            return_full_output=self.return_full_output,
            verbose = False
        )
        measurement_responses = ctxlm.predict(messages)

        measured_data = []
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

        return measured_data
    

    def _measure_vllm(self):
        """
        Extracts measurements from the text chunks for the identified items.

        Args:

        Returns:
            
        """
        messages = []
        for i, datapoint in enumerate(self.data):
            item = {k: v for k,v in datapoint.items() if k not in ['context', 'chunk_id','paper_id', 'measurement']}
            measurement = datapoint['measurement']
            instructions = (
                f"You are an expert in extracting precise numerical data from user provided, scientific text. "
                f"A value is a single numerical measurement explicitly mentioned in the context. "
                f"You will be queried with a description of an specific entity to be measured, along with the measurement type to report for. "
                f"Your task is to extract the corresponding value from the provided context. "
                f"Copy the value exactly as it appears in the context. "
                f"Give the value only, and do not include any units of measurement, descriptors, or explanation in your response. "
                f"Respond 'None' if the requested information is not explicitly available in the given context."
            )
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
        messages = []
        message_data_ids = []
        sampling_params = []
        for i, datapoint in enumerate(self.data):
            item = {k: v for k,v in datapoint.items() if k not in ['context', 'chunk_id', 'paper_id', 'measurement', 'value', 'context_scores', 'parametric_scores', 'copying_scores', 'linear_probes']}
            measurement = datapoint['measurement']
            measurement_val = datapoint['value']

            measurement_description = self.measurement_schema.model_fields[measurement].description
            available_units = self.measurement_schema.model_fields[measurement].json_schema_extra.get('units', None)

            if available_units is not None:
                units_list = available_units + ['other']
                units_str = ', '.join(units_list)
                instructions = (
                    f"You are an expert in data collection and scientific measurements. "
                    f"You will be given context from a research paper, along with a description of a measurement value and the entity it was reported for. "
                    f"Your task is to determine the unit of measurement for that data point by referencing the context, and then choosing from a list of available options. "
                    f"To ensure units follow standard formatting conventions, your response should be limited to options from among the given list. "
                    f"If none of the options fit with what is seen in the context, respond with the unit 'other'. "
                    f"Your response should include the unit only, do not include any additional explanation or text.\n\n"
                )
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
                guided_decoding_params = GuidedDecodingParams(
                    choice = units_list
                )
                params = SamplingParams(
                    **self.sampling_params,
                    guided_decoding=guided_decoding_params
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
            item = {k: v for k,v in datapoint.items() if k not in ['context', 'chunk_id', 'paper_id', 'context_scores', 'parametric_scores', 'copying_scores', 'linear_probes']}
            instructions = (
                f"You are an expert in discerning whether or not a given data point is textually accurate. "
                f"You will be given a context from a research paper, along with a data point that is assumed to be extracted from it. "
                f"Your task is to determine if the data point is supported by evidence in the context. "
                f"Respond 'true' only if the context explicity provides evidence for the data point. "
                f"Respond 'false' otherwise."
            )
            context = datapoint['context']
            query = f"Does the given context support the data point: {item}?"
            prompt = (
            f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
            )
            messages.append([
                {"role": "user","content": prompt}]
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
            for j in range(len(chunks[i])):
                self.data.append({'paper_id': i, 'chunk_id': j, 'context' : chunks[i][j]})

        self.data = self._filter()
        self.data = self._identify()
        self.data = self._measurements_filter()
        if self.return_full_output:
            self.data = self._measure()
        else:
            self.data = self._measure_vllm()
        self.data = self._standardize()
        self.data = self._judge()

        return self.data
    

    def save(self, filepath: str):
        """
        Saves the measurement data to a csv.

        Args:
            filepath (str): The path to the file where the data will be saved.
        """
        df = pd.DataFrame(self.data)
        df.to_csv(filepath, index=False)



