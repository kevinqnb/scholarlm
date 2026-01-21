from pydantic import BaseModel
import pandas as pd
import json
import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


class JudgementLM:
    """
    A language model class designed for judging extracted responses, by classifying them 
    as 'hallucination', 'disorientation', 'deviation', or 'valid' based on provided context.

    Args:
        model_name (str): The name or path of the pre-trained language model from the huggingface 
            collection.
        sampling_params (dict[str, any]): A dictionary of sampling parameters for text generation.

    Attributes:

    """
    def __init__(
        self,
        model_name: str,
        identification_schema: BaseModel,
        measurement_schema: BaseModel,
        sampling_params: dict[str, any] = {},
    ):
        self.model_name = model_name
        self.identification_schema = identification_schema
        self.measurement_schema = measurement_schema
        self.sampling_params = {
            "temperature" : 0.90,
            "top_p" : 0.95,
            "top_k" : 64,
            "repetition_penalty" : 1.0,
            "max_tokens" : 2048,
        } | sampling_params

        self.entity_description = identification_schema.model_config['entity_description']
        self.primary_identifier = identification_schema.model_config['primary_identifier']

        self.llm = LLM(model=model_name)


    def _hallucination(self):
        """
        Internal method to judge hallucination in responses.

        Returns:
            (Fill in)
        """
        instructions = (
            f"You are an expert in validating results from data extraction tasks. "
            f"You will be given context from a scientific research paper, along with a data point that "
            f"a large language model has generated upon being prompted to extract relevant data. "
            f"Your task is to determine whether or not the extracted data point explicity appears within the provided context. "
            f"Respond with 'false' if the extracted data point's 'value' feature does not explicity appear within the context, indicating a hallucination. "
            f"Respond with 'true' only if the extracted data point's 'value' feature explicity appears somewhere within the context. "
            f"Only respond with 'true' or 'false'. Do not include any other text or explanation in your response."
        )

        messages = []
        message_ids = []
        for i, entry in enumerate(self.data):
            if 'judgement' in entry:
                continue

            context = entry.get('context', None)

            # Only use the value when checking for hallucinations:
            datapoint = {
                "value": entry['value'],
            }

            prompt = (
                f'## Instructions:\n'
                f"{instructions}\n\n"
                f"## Context:\n"
                f"{context}\n\n"
                f"## Extracted Data Point:\n"
                f"{json.dumps(datapoint)}\n\n"
                f"## Query:\n"
                f"Does the extracted data point appear explicity within the context?"
            )

            messages.append([
                {"role": "user", "content": prompt}]
            )
            message_ids.append(i)

        guided_decoding_params = GuidedDecodingParams(
            choice = ['true', 'false']
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )

        responses = self.llm.chat(messages = messages, sampling_params = sampling_params)
        response_texts = [r.outputs[0].text for r in responses]

        filtered_data = [d for d in self.data]
        for i, resp in enumerate(response_texts):
            idx = message_ids[i]
            if resp == 'false':
                filtered_data[idx]['judgement'] = 'hallucination'
        
        return filtered_data


    def _disorientation(self):
        """
        Internal method to judge disorientation in responses.

        Returns:
            (Fill in)
        """
        instructions = (
            f"You are an expert in validating results from data extraction tasks. "
            f"You will be given context from a scientific research paper, along with a data point that "
            f"a large language model has generated upon being prompted to extract relevant data. "
            f"Your task is to use the context to determine whether or not the extracted data point's 'value' feature "
            f"is being correctly associated with the rest of the data point's entity attributes (such as name, measurement type, date, location, etc.). "
            f"Respond with 'false' if the extracted data point's 'value' feature is incorrectly attributed to the given entity. "
            f"Respond with 'true' only if the extracted data point's 'value' feature is explificity associated with the given entity within the context. "
            f"Only respond with 'true' or 'false'. Do not include any other text or explanation in your response."
        )

        messages = []
        message_ids = []
        for i, entry in enumerate(self.data):
            if 'judgement' in entry:
                continue

            context = entry.get('context', None)

            entity = {
                f: entry[f] for f in self.identification_schema.model_fields.keys()
            }
            measurement = {
                'measurement': self.measurement_schema.model_fields[entry['measurement']].description,
                'value': entry['value']
            }
            measurement_names = entry.get('measurement_names', [])
            entity_names = entry.get('entity_names', [])

            prompt = (
                f'## Instructions:\n'
                f"{instructions}\n\n"
                f"## Context:\n"
                f"{context}\n\n"
                f"## Extracted Data Point:\n"
                f"Entity:\n"
                f"{json.dumps(entity)}\n\n"
                f"Measurement:\n"
                f"{json.dumps(measurement)}\n\n"
                f"## Query:\n"
                f"Is the extracted measurement value correctly attributed to the given entity?"
                f"Note that the entity may be referred to by any of the following extended names or abbreviations: {entity_names}. "
                f"Also note that the text may also refer to the measurement using any of the following abbreviations: {measurement_names}."
            )

            messages.append([
                {"role": "user", "content": prompt}]
            )
            message_ids.append(i)

        guided_decoding_params = GuidedDecodingParams(
            choice = ['true', 'false']
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )

        responses = self.llm.chat(messages = messages, sampling_params = sampling_params)
        response_texts = [r.outputs[0].text for r in responses]

        filtered_data = [d for d in self.data]
        for i, resp in enumerate(response_texts):
            idx = message_ids[i]
            if resp == 'false':
                filtered_data[idx]['judgement'] = 'disorientation'
        
        return filtered_data
    

    def _deviation(self):
        """
        Internal method to judge deviation in responses.

        Returns:
            (Fill in)
        """
        instructions = (
            f"You are an expert in validating results from data extraction tasks. "
            f"You will be given context from a scientific research paper, along with a data point that "
            f"a large language model has generated upon being prompted to extract relevant data. "
            f"Your task is to use the context to determine whether or not the extracted data point has deviated from its given instructions. "
            f"Specifically, we say that a deviation has occured if the data point's 'value' feature "
            f"corresponds to a range of values, inequality, non-numerical description, "
            f"or is a measurement for a general collection of entities rather than a direct numerical measurement for a single, specific entity. "
            f"Respond with 'true' if the extracted data point's 'value' feature is one of these types, indicating a deviation from instructions. "
            f"Respond with 'false' only if the extracted data point's 'value' feature is a direct numerical measurement for a single, specific entity. "
            f"Only respond with 'true' or 'false'. Do not include any other text or explanation in your response."
        )

        messages = []
        message_ids = []
        for i, entry in enumerate(self.data):
            if 'judgement' in entry:
                continue

            context = entry.get('context', None)

            datapoint = {
                f: entry[f] for f in self.identification_schema.model_fields.keys()
            }
            datapoint['measurement'] = self.measurement_schema.model_fields[entry['measurement']].description
            datapoint['value'] = entry['value']
            measurement_names = entry.get('measurement_names', [])
            entity_names = entry.get('entity_names', [])

            prompt = (
                f'## Instructions:\n'
                f"{instructions}\n\n"
                f"## Context:\n"
                f"{context}\n\n"
                f"## Extracted Data Point:\n"
                f"{json.dumps(datapoint)}\n\n"
                f"## Query:\n"
                f"Does the extracted data point deviate from its given intstructions?",
                f"Note that the entity may be referred to by any of the following extended names or abbreviations: {entity_names}. "
                f"Also note that the text may also refer to the measurement using any of the following abbreviations: {measurement_names}."
            )

            messages.append([
                {"role": "user", "content": prompt}]
            )
            message_ids.append(i)

        guided_decoding_params = GuidedDecodingParams(
            choice = ['true', 'false']
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )

        responses = self.llm.chat(messages = messages, sampling_params = sampling_params)
        response_texts = [r.outputs[0].text for r in responses]

        filtered_data = [d for d in self.data]
        for i, resp in enumerate(response_texts):
            idx = message_ids[i]
            if resp == 'true':
                filtered_data[idx]['judgement'] = 'deviation'
        
        return filtered_data


    def fit(
        self,
        data: list[dict[str, any]]
    ) -> list[dict[str, any]]:
        """
        Fit the JudgementLM model on the provided data.

        Args:
            data (list[dict[str, any]]): A list of dictionaries containing response data.

        Returns:
            list[dict[str, any]]: The input data augmented with judgement results.
        """
        self.data = data
        self.data = self._hallucination()
        self.data = self._disorientation()
        self.data = self._deviation()

        for point in self.data:
            if point.get('judgement', None) is None:
                point['judgement'] = 'valid'

        return self.data
