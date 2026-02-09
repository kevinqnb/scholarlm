from pydantic import BaseModel
import numpy as np
import pandas as pd
import re
from io import StringIO
import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from .contextlm import ContextLM
from .instruction_prompts import (
    IDENTIFY_FEATURE_TERMS_INSTRUCTIONS,
    IDENTIFY_FEATURE_UNITS_INSTRUCTIONS,
    IDENTIFY_ENTITY_FEATURE_PAIRS_INSTRUCTIONS,
    PAGE_LOCATE_INSTRUCTIONS,
    TABLE_LOCATE_INSTRUCTIONS,
    MEASURE_VALUE_INSTRUCTIONS,
    MEASURE_TABLE_ROW_INSTRUCTIONS,
    MEASURE_TABLE_COLUMN_INSTRUCTIONS,
    STANDARDIZE_MEASUREMENTS_INSTRUCTIONS,
    STANDARDIZE_UNITS_INSTRUCTIONS,
)


def response_validator(response_structure, response):
    pyd = response_structure.model_validate_json(response)
    out_dict = pyd.model_dump()
    return out_dict


class ListResponse(BaseModel):
    items: list[str]


class BooleanDecisionResponse(BaseModel):
    """Structured response that encourages reasoning before a boolean decision."""
    explanation: str
    answer: bool


class MeasurementLM:
    """
    A language model class designed for organized collection of measurements from scientific text.

    Args:
        model_name (str): The name or path of the pre-trained language model from the huggingface 
            collection.
        sampling_params (dict[str, any]): A dictionary of sampling parameters for text generation.
        entity_identification_prompt (str): The prompt template for entity identification.
        entity_identification_schema (BaseModel): The pydantic schema for entity identification.
        feature_info_dict (dict[str, any]): A dictionary containing information about the features to be
            measured.

    Attributes:

    """
    def __init__(
        self,
        model_name: str,
        entity_identification_prompt: str,
        entity_identification_schema: BaseModel,
        feature_info_dict: dict[str, any],
        sampling_params: dict[str, any] = {},
    ):
        self.model_name = model_name
        self.sampling_params = {
            "temperature" : 0.90,
            "top_p" : 0.95,
            "top_k" : 64,
            "repetition_penalty" : 1.0,
            "max_tokens" : 2048,
        } | sampling_params
        self.entity_identification_prompt = entity_identification_prompt
        self.entity_identification_schema = entity_identification_schema
        self.feature_info_dict = feature_info_dict
        self.llm = LLM(model=model_name)


    def _identify_feature_terms(self):
        """
        Identifies terms used to describe features in the text.

        Args:
            
        Returns:
            
        """
        instructions = IDENTIFY_FEATURE_TERMS_INSTRUCTIONS
        messages = []
        message_ids = []
        for i, datapoint in enumerate(self.data):
            context = datapoint['context']
            for feature in self.feature_info_dict:
                feature_description = self.feature_info_dict[feature]['description']

                query = (
                    f"Feature description: {feature_description}\n\n"
                    f"What terms are used to refer to the described feature in the context?"
                )
                prompt = (
                    f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
                )
                messages.append([
                    {"role": "user", "content": prompt}
                ])
                message_ids.append((i, feature))

        guided_decoding_params = GuidedDecodingParams(
            json=ListResponse.model_json_schema()
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )
        responses = self.llm.chat(messages = messages, sampling_params = sampling_params)
        response_texts = [r.outputs[0].text for r in responses]
        feature_term_data = [d for d in self.data]
        for i, resp in enumerate(response_texts):
            idx, feature = message_ids[i]
            try:
                alt_names = response_validator(ListResponse, resp)['items']
                if feature_term_data[idx].get('feature_terms') is None:
                    feature_term_data[idx]['feature_terms'] = {}
                feature_term_data[idx]['feature_terms'][feature] = alt_names
            except:
                print("Error parsing alternative names response.")
                if feature_term_data[idx].get('feature_terms') is None:
                    feature_term_data[idx]['feature_terms'] = {}
                feature_term_data[idx]['feature_terms'][feature] = []

        return feature_term_data


    def _identify_feature_units(self):
        """
        Identifies units used to measure features in the text.

        Args:
            
        Returns:

        """
        instructions = IDENTIFY_FEATURE_UNITS_INSTRUCTIONS
        messages = []
        message_ids = []
        sampling_params = []
        for i, datapoint in enumerate(self.data):
            context = datapoint['context']
            doc_id = datapoint.get('document_id', None)
            for feature in self.feature_info_dict:
                feature_description = self.feature_info_dict[feature]['description']
                feature_terms = datapoint.get('feature_terms', {}).get(feature, [])
                feature_units = self.feature_info_dict[feature].get('units', None)

                if len(feature_units) == 0:
                    continue

                query = (
                    f"Feature description: {feature_description}\n"
                    f"Terminology used for the feature: {feature_terms}\n\n"
                    f"What units does the context use to measure described feature, if any? Choose from among given options: {feature_units}."
                )
                prompt = (
                    f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
                )
                messages.append([
                    {"role": "user", "content": prompt}
                ])
                message_ids.append((i, feature))

                '''
                guided_decoding_params = GuidedDecodingParams(
                    choice = feature_units
                )
                params = SamplingParams(
                    **self.sampling_params,
                    guided_decoding=guided_decoding_params
                )
                sampling_params.append(params)
                '''


        sampling_params = SamplingParams(
            **self.sampling_params,
        )
        responses = self.llm.chat(messages = messages, sampling_params = sampling_params)
        response_texts = [r.outputs[0].text for r in responses]
        feature_unit_data = [d for d in self.data]
        for i, resp in enumerate(response_texts):
            idx, feature = message_ids[i]
            if resp.strip().lower() != 'none':
                if feature_unit_data[idx].get('units') is None:
                    feature_unit_data[idx]['units'] = {}
                feature_unit_data[idx]['units'][feature] = resp.strip()
            else:
                if feature_unit_data[idx].get('units') is None:
                    feature_unit_data[idx]['units'] = {}
                feature_unit_data[idx]['units'][feature] = None

        return feature_unit_data


    def _identify_entities(self):
        """
        Identifies entities in the text based on the identification schema.

        Args:
            
        Returns:
            
        """
        from pydantic import create_model

        IdentificationList = create_model(
            "IdentificationList",
            items=(list[self.entity_identification_schema], ...),
        )
        identification_list_json = IdentificationList.model_json_schema()

        messages = []
        for i, datapoint in enumerate(self.data):
            instructions = self.entity_identification_prompt
            context = datapoint['context']

            query = "Follow the instructions to identify the items mentioned in the context."
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
        entity_data = []
        for i, r in enumerate(response_texts):
            try:
                resp_validated = response_validator(IdentificationList, r)
            except:
                print("Validation error in identification response.")
                resp_validated = {'items': []}

            for entity in resp_validated['items']:
                entity_data.append(
                    self.data[i] | entity
                )

        return entity_data

    
    def _identify_entity_feature_pairs(self):
        """
        Identifies measurements in the text based on the measurement schema.

        Args:
            
        Returns:
            
        """
        instructions = IDENTIFY_ENTITY_FEATURE_PAIRS_INSTRUCTIONS
        messages = []
        message_ids = []
        for i, datapoint in enumerate(self.data):
            context = datapoint['context']
            doc_id = datapoint.get('document_id', None)
            entity_description = {
                k: v for k,v in datapoint.items() if k in self.entity_identification_schema.model_fields.keys()
            }
            for feature in self.feature_info_dict:
                feature_description = self.feature_info_dict[feature].get('description', '')
                feature_terms = datapoint.get('feature_terms', {}).get(feature, [])

                query = (
                    f"Feature description: {feature_description}\n"
                    f"Terminology used for the feature: {feature_terms}\n"
                    f"Entity description: {entity_description}\n\n"
                    f"Does the context provide data for measuring the described feature in reference to the given entity?\n\n"
                )
                prompt = (
                    f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
                )
                messages.append([
                    {"role": "user", "content": prompt}
                ])
                message_ids.append((i, feature))

        guided_decoding_params = GuidedDecodingParams(
            json=BooleanDecisionResponse.model_json_schema()
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )
        responses = self.llm.chat(messages = messages, sampling_params = sampling_params)
        response_texts = [r.outputs[0].text for r in responses]

        identified_data = []
        for i, resp in enumerate(response_texts):
            idx, feature = message_ids[i]
            try:
                decision = response_validator(BooleanDecisionResponse, resp)
                answer = bool(decision.get('answer', False))
            except:
                print("Validation error in entity-feature decision response.")
                answer = False

            if answer:
                datapoint = self.data[idx]
                feature_terms = datapoint.get('feature_terms', {}).get(feature, [])
                units = datapoint.get('units', {}).get(feature, None)
                doc_id = datapoint.get('document_id', None)
                identified_data.append(
                    datapoint | {
                        'feature': feature,
                        'feature_terms': feature_terms,
                        'units': units,
                    }
                )

        return identified_data
    

    def _page_locate(self):
        """
        Locates page numbers, for the identified measurements.

        Args:

        Returns:
            
        """
        instructions = PAGE_LOCATE_INSTRUCTIONS
        fields = list(self.entity_identification_schema.model_fields.keys())
        messages = []
        message_ids = []
        for i, datapoint in enumerate(self.data):
            context = datapoint['context']
            feature = datapoint.get('feature')
            feature_description = self.feature_info_dict[feature]['description']
            feature_terms = datapoint.get('feature_terms', [])
            entity_description = {k: v for k,v in datapoint.items() if k in fields}

            pages = re.findall(r'<page number="(\d+)">', context)
            pages = [int(p) for p in pages]
            for p in pages:
                page_start = context.find(f'<page number="{p}">') + len(f'<page number="{p}">')
                page_end = context.find(f'</page>', page_start)
                page_text = context[page_start: page_end].strip()

                query = (
                    f"Feature description: {feature_description}\n"
                    f"Terminology used for the feature: {feature_terms}\n"
                    f"Entity description: {entity_description}\n\n"
                    f"Does the given page contain a measurement for the given feature and entity?\n\n"
                )
                prompt = (
                    f"## Instructions:\n{instructions}\n\n## Context:\n{page_text}\n\n## Query:\n{query}"
                    )
                messages.append(
                    [{"role": "user","content": prompt}]
                )
                message_ids.append((i, p))


        guided_decoding_params = GuidedDecodingParams(
            json=BooleanDecisionResponse.model_json_schema()
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params,
            logprobs = 1,
        )

        responses = self.llm.chat(
            messages = messages,
            sampling_params = sampling_params,
        )
        response_texts = [r.outputs[0].text for r in responses]
        response_probs = [r.outputs[0].cumulative_logprob for r in responses]

        page_id_data = []
        for i, resp in enumerate(response_texts):
            try:
                decision = response_validator(BooleanDecisionResponse, resp)
                answer = bool(decision.get('answer', False))
            except:
                print("Validation error in page-locate decision response.")
                answer = False

            if answer:
                idx, page_number = message_ids[i]
                datapoint = self.data[idx]
                context = datapoint['context']
                page_start = context.find(f'<page number="{page_number}">') + len(f'<page number="{page_number}">')
                page_end = context.find(f'</page>', page_start)
                page_text = context[page_start: page_end].strip()

                page_id_data.append(
                    datapoint | 
                    {'context': page_text, 'measurement_id': i} |
                    {'page_number': page_number} |
                    {'page_logprob': response_probs[i]}
                )

        return page_id_data
    

    def _table_locate(self):
        """
        Locates page numbers, for the identified measurements.

        Args:

        Returns:
            
        """
        instructions = TABLE_LOCATE_INSTRUCTIONS
        fields = list(self.entity_identification_schema.model_fields.keys())
        messages = []
        message_tuples = []
        message_ids = []
        for i, datapoint in enumerate(self.data):
            context = datapoint['context']
            feature = datapoint.get('feature')
            feature_description = self.feature_info_dict[feature]['description']
            feature_terms = datapoint.get('feature_terms', [])
            entity_description = {k: v for k,v in datapoint.items() if k in fields}

            tables = re.findall(r'<table number="(\d+)">', context)
            tables = [int(t) for t in tables]

            for t in tables:
                table_start = context.find(f'<table number="{t}">') + len(f'<table number="{t}">')
                table_end = context.find(f'</table>', table_start)
                table_text = context[table_start: table_end].strip()

                query = (
                    f"Feature description: {feature_description}\n"
                    f"Terminology used for the feature: {feature_terms}\n"
                    f"Entity description: {entity_description}\n\n"
                    f"Does the table contain a measurement for the given feature and entity?\n\n"
                )
                prompt = (
                    f"## Instructions:\n{instructions}\n\n## Context:\n{table_text}\n\n## Query:\n{query}"
                    )
                messages.append(
                    [{"role": "user","content": prompt}]
                )
                message_tuples.append((instructions, table_text, query))
                message_ids.append((i, t))


        guided_decoding_params = GuidedDecodingParams(
            json=BooleanDecisionResponse.model_json_schema()
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params,
            logprobs = 1,
        )

        responses = self.llm.chat(
            messages = messages,
            sampling_params = sampling_params,
        )
        response_texts = [r.outputs[0].text for r in responses]
        response_probs = [r.outputs[0].cumulative_logprob for r in responses]
        table_id_data = [
            d | {'table_number': -1, 'table_logprob': 0.0} for d in self.data
        ]
        for i, resp in enumerate(response_texts):
            try:
                decision = response_validator(BooleanDecisionResponse, resp)
                answer = bool(decision.get('answer', False))
            except:
                print("Validation error in table-locate decision response.")
                answer = False

            if answer:
                idx, table_number = message_ids[i]
                datapoint = self.data[idx]
                table_id_data[idx] = datapoint | {
                    'table_number': table_number, 
                    'table_logprob': response_probs[i]
                }

        return table_id_data


    def _measure_vllm(self):
        """
        Extracts measurements from the text chunks for the identified items.

        Args:

        Returns:
            
        """
        instructions = MEASURE_VALUE_INSTRUCTIONS
        fields = list(self.entity_identification_schema.model_fields.keys())
        messages = []
        message_ids = []
        for i, datapoint in enumerate(self.data):
            if datapoint['table_number'] == -1:
                context = datapoint['context']
                feature = datapoint.get('feature')
                feature_description = self.feature_info_dict[feature]['description']
                feature_terms = datapoint.get('feature_terms', [])
                entity_description = {k: v for k,v in datapoint.items() if k in fields}

                query = (
                    f"Feature description: {feature_description}\n"
                    f"Terminology used for the feature: {feature_terms}\n"
                    f"Entity description: {entity_description}\n\n"
                    f"Extract the value reported by the context for the given feature and entity."
                )
                prompt = (
                    f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
                    )
                messages.append([
                    {"role": "user","content": prompt}]
                )
                message_ids.append(i)

        sampling_params = SamplingParams(
            **self.sampling_params
        )

        responses = self.llm.chat(messages = messages, sampling_params = sampling_params)
        response_texts = [r.outputs[0].text for r in responses]
        measured_data = []
        for i, resp in enumerate(response_texts):
            if resp.strip().lower() != 'none':
                idx = message_ids[i]
                measured_data.append(self.data[idx] | {'value': resp})

        measured_data = measured_data + [d for d in self.data if d['table_number'] != -1]

        return measured_data

    
    def _measure_vllm_rows(self):
        """
        Extracts measurements from tables in the text.

        Args:

        Returns:
            
        """
        # Next, extract the unique row name necessary to locate the measurement:
        instructions = MEASURE_TABLE_ROW_INSTRUCTIONS
        fields = list(self.entity_identification_schema.model_fields.keys())
        messages = []
        sampling_params = []
        message_ids = []
        for i, datapoint in enumerate(self.data):
            if datapoint['table_number'] != -1:
                context = datapoint['context']
                table_number = datapoint['table_number']
                table_start = f'<table number="{table_number}">' 
                table_start = context.find(f'<table number="{table_number}">')
                table_end = context.find(f'</table>', table_start) + len(f'</table>')
                table_text = context[table_start: table_end].strip()
                if not table_text:
                    continue
                tables = pd.read_html(StringIO(table_text))
                table_df = tables[0]
                row_names = table_df.loc[:,"index"].to_list() if 'index' in table_df.columns else []
                row_names = [str(name) for name in row_names]

                feature = datapoint.get('feature')
                feature_description = self.feature_info_dict[feature]['description']
                feature_terms = datapoint.get('feature_terms', [])
                entity_description = {k: v for k,v in datapoint.items() if k in fields}

                query = (
                    f"Feature description: {feature_description}\n"
                    f"Terminology used for the feature: {feature_terms}\n"
                    f"Entity description: {entity_description}\n\n"
                    f"Extract the row index name in table {table_number} necessary to locate the measurement for the given feature and entity."
                )
                prompt = (
                    f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
                )
                messages.append([{"role": "user","content": prompt}])
                guided_decoding_params = GuidedDecodingParams(
                    choice = row_names + ['None']
                )
                params = SamplingParams(
                    **self.sampling_params,
                    guided_decoding=guided_decoding_params
                )
                sampling_params.append(params)
                message_ids.append(i)

        
        responses = self.llm.chat(
            messages = messages,
            sampling_params = sampling_params,
        )
        response_texts = [r.outputs[0].text for r in responses]
        measured_data = [d for d in self.data]
        for i, resp in enumerate(response_texts):
            idx = message_ids[i]
            datapoint = self.data[idx]
            if resp.strip().lower() != 'none':
                measured_data[idx] = datapoint | {
                    'row_index': resp
                }

        return measured_data

    
    def _measure_vllm_columns(self):
        """
        Extracts measurements from tables in the text.

        Args:

        Returns:
            
        """
        instructions = MEASURE_TABLE_COLUMN_INSTRUCTIONS
        fields = list(self.entity_identification_schema.model_fields.keys())
        messages = []
        sampling_params = []
        message_ids = []
        for i, datapoint in enumerate(self.data):
            if int(datapoint['table_number']) != -1:
                context = datapoint['context']
                table_number = int(datapoint['table_number'])
                table_start = f'<table number="{table_number}">' 
                table_start = context.find(f'<table number="{table_number}">')
                table_end = context.find(f'</table>', table_start) + len(f'</table>')
                table_text = context[table_start: table_end].strip()
                tables = pd.read_html(StringIO(table_text))
                table_df = tables[0]
                column_names = [str(name) for name in table_df.columns.tolist()]
                column_names = [str(name) for name in column_names if str(name) != 'index']

                feature = datapoint.get('feature')
                feature_description = self.feature_info_dict[feature]['description']
                feature_terms = datapoint.get('feature_terms', [])
                entity_description = {k: v for k,v in datapoint.items() if k in fields}
                
                query = (
                    f"Feature description: {feature_description}\n"
                    f"Terminology used for the feature: {feature_terms}\n"
                    f"Entity description: {entity_description}\n\n"
                    f"Extract the column index name in table {table_number} necessary to locate the measurement for the given feature and entity."
                )
                prompt = (
                    f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
                )
                messages.append([
                    {"role": "user","content": prompt}]
                )
                guided_decoding_params = GuidedDecodingParams(
                    choice = column_names + ['None']
                )
                params = SamplingParams(
                    **self.sampling_params,
                    guided_decoding=guided_decoding_params
                )
                sampling_params.append(params)
                message_ids.append(i)


        responses = self.llm.chat(
            messages = messages,
            sampling_params = sampling_params,
        )
        response_texts = [r.outputs[0].text for r in responses]
        measured_data = [d for d in self.data]
        for i, resp in enumerate(response_texts):
            idx = message_ids[i]
            datapoint = self.data[idx]
            if resp.strip().lower() != 'none':
                measured_data[idx] = datapoint | {
                    'column_index': resp
                }

        return measured_data


    def _table_extract(self):
        """
        Extracts measurements from tables in the text.
        """
        table_extracted_data = [d for d in self.data]
        for i, datapoint in enumerate(self.data):
            if datapoint.get('row_index', None) is not None and datapoint.get('column_index', None) is not None:
                context = datapoint['context']
                table_number = int(datapoint['table_number'])
                table_start = context.find(f'<table number="{table_number}">')
                table_end = context.find(f'</table>', table_start) + len(f'</table>')
                table_text = context[table_start: table_end].strip()
                tables = pd.read_html(StringIO(table_text))
                table_df = tables[0]

                col_name = datapoint['column_index']
                row_name = datapoint['row_index']

                # Get matching rows
                matched_rows = table_df.loc[table_df["index"] == row_name][col_name]
                
                # Handle edge cases: no matches, multiple matches, or single match
                if len(matched_rows) == 0:
                    # No matching row found
                    print("No matching row found in table extraction.")
                    val = None
                elif len(matched_rows) == 1:
                    val = matched_rows.item()
                else:
                    # Multiple matches - take the first one
                    print("Multiple matching rows found in table extraction, taking the first match.")
                    val = matched_rows.iloc[0]
                
                table_extracted_data[i] = datapoint | {'value': val}

        return [t for t in table_extracted_data if t.get('value', None) is not None]


    def _standardize_measurements(self):
        """
        Standardizes the measurement units for the extracted measurements.

        Args:

        Returns:
            
        """
        instructions = STANDARDIZE_MEASUREMENTS_INSTRUCTIONS
        fields = list(self.entity_identification_schema.model_fields.keys())
        messages = []
        message_data_ids = []
        sampling_params = []
        for i, datapoint in enumerate(self.data):
            context = datapoint['context']
            feature = datapoint.get('feature')
            feature_description = self.feature_info_dict[feature]['description']
            feature_terms = datapoint.get('feature_terms', [])
            entity_description = {k: v for k,v in datapoint.items() if k in fields}
            measurement_val = datapoint['value']

            query = (
                f"Feature description: {feature_description}\n"
                f"Terminology used for the feature: {feature_terms}\n"
                f"Entity description: {entity_description}\n"
                f"Extracted measurement: {measurement_val}\n\n"
                f"Standardize the measurement value for the given data point. "
            )
            prompt = (
                f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
            )
            messages.append(
                [{"role": "user", "content": prompt}]
            )
            message_data_ids.append(i)

            params = SamplingParams(
                **self.sampling_params,
            )
            sampling_params.append(params)

        responses = self.llm.chat(messages = messages, sampling_params = sampling_params)
        response_units = [r.outputs[0].text for r in responses]
        
        standardized_data = [datapoint for datapoint in self.data]
        for i, resp in enumerate(response_units):
            standardized_data[message_data_ids[i]]['value'] = resp.strip()

        return standardized_data


    def fit(
        self,
        documents : list[str],
    ):
        """
        Fits the MeasurementLM to the provided text chunks by filtering, identifying items, 
        and extracting measurements.

        Args:
            documents (list[str]): A list of text documents.
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
        for i, doc in enumerate(documents):
            self.data.append({'document_id': i, 'context' : doc})

        self.data = self._identify_feature_terms()
        self.data = self._identify_feature_units()
        self.data = self._identify_entities()
        self.data = self._identify_entity_feature_pairs()
        self.data = self._page_locate()
        self.data = self._table_locate()
        self.data = self._measure_vllm()
        self.data = self._measure_vllm_rows()
        self.data = self._measure_vllm_columns()
        self.data = self._table_extract()
        self.data = self._standardize_measurements()

        return self.data


    def save(self, filepath: str):
        """
        Saves the measurement data to a csv.

        Args:
            filepath (str): The path to the file where the data will be saved.
        """
        df = pd.DataFrame(self.data)
        df.to_csv(filepath, index=False)



