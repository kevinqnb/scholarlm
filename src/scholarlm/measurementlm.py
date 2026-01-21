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
    IDENTIFY_MEASUREMENTS_RELEVANCE_INSTRUCTIONS,
    IDENTIFY_MEASUREMENT_TERMS_INSTRUCTIONS,
    IDENTIFY_ENTITY_ALIASES_INSTRUCTIONS,
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
        probe: bool = False,
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
        self.probe = probe

        if not self.probe:
            self.llm = LLM(model=model_name, tensor_parallel_size=2)
        else:
            ctxlm_params = {
                k: v for k,v in sampling_params.items() if k not in [
                    'max_tokens', 'seed', 'temperature', 'stop'
                ]
            }
            ctxlm_params['do_sample'] = False
            ctxlm_params['max_new_tokens'] = 1
            self.ctxlm = ContextLM(
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                sampling_params=ctxlm_params,
                nnsight_kwargs = {"torch_dtype": torch.bfloat16},
            )
        

    
    def _identify_measurements(self):
        """
        Identifies measurements in the text based on the measurement schema.

        Args:
            
        Returns:
            
        """
        instructions = IDENTIFY_MEASUREMENTS_RELEVANCE_INSTRUCTIONS
        messages = []
        message_ids = []
        for i, datapoint in enumerate(self.data):
            for measurement in self.measurement_schema.model_fields.keys():
                measurement_description = self.measurement_schema.model_fields[measurement].description

                context = datapoint['context']
                query = (
                    f"Is the context relevant to measuring or identifying "
                    f"{measurement_description} for {self.entity_description}?"
                )
                prompt = (
                    f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
                )
                messages.append([
                    {"role": "user", "content": prompt}
                ])
                message_ids.append((i, measurement))

        guided_decoding_params = GuidedDecodingParams(
            choice = ['true', 'false']
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )
        responses = self.llm.chat(messages = messages, sampling_params = sampling_params)
        response_texts = [r.outputs[0].text for r in responses]

        instructions2 = IDENTIFY_MEASUREMENT_TERMS_INSTRUCTIONS
        messages2 = []
        message_ids2 = []
        for i, resp in enumerate(response_texts):
            idx, measurement = message_ids[i]
            if resp == 'true':
                datapoint = self.data[idx]
                context = datapoint['context']
                measurement_description = self.measurement_schema.model_fields[measurement].description
                query = (
                    f"Extract any terms that refer to "
                    f"{measurement_description} in the given context. "
                )
                prompt = (
                    f"## Instructions:\n{instructions2}\n\n## Context:\n{context}\n\n## Query:\n{query}"
                )
                messages2.append([
                    {"role": "user", "content": prompt}]
                )
                message_ids2.append((idx, measurement))

        guided_decoding_params = GuidedDecodingParams(
            json=ListResponse.model_json_schema()
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )
        responses = self.llm.chat(messages = messages2, sampling_params = sampling_params)
        response_texts = [r.outputs[0].text for r in responses]

        identified_data_by_idx = {}
        for i, resp in enumerate(response_texts):
            idx, measurement = message_ids2[i]
            datapoint = self.data[idx]
            try:
                alt_names = response_validator(ListResponse, resp)['items']
                if idx not in identified_data_by_idx:
                    identified_data_by_idx[idx] = {measurement: alt_names}
                else:
                    identified_data_by_idx[idx][measurement] = alt_names
            except:
                print("Error parsing alternative names response.")
                identified_data_by_idx[idx] = {measurement: []}

        identified_data = []
        for idx in identified_data_by_idx:
            identified_data.append(
                self.data[idx] | {'measurement_names': identified_data_by_idx[idx]}
            )

        return identified_data
    

    def _identify_entities(self):
        """
        Identifies entities in the text based on the identification schema.

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


        instructions2 = IDENTIFY_ENTITY_ALIASES_INSTRUCTIONS
        messages2 = []
        message_ids2 = []
        for i, datapoint in enumerate(entity_data):
            context = datapoint['context']
            item = {k: v for k,v in datapoint.items() if k in self.identification_schema.model_fields.keys()}
            query = (
                f"Extract any abbreviations or extended names that refer to "
                f"{item} in the given context. "
            )
            prompt = (
                f"## Instructions:\n{instructions2}\n\n## Context:\n{context}\n\n## Query:\n{query}"
            )
            messages2.append([
                {"role": "user", "content": prompt}]
            )
            message_ids2.append(i)

        guided_decoding_params = GuidedDecodingParams(
            json=ListResponse.model_json_schema()
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )
        responses = self.llm.chat(messages = messages2, sampling_params = sampling_params)
        response_texts = [r.outputs[0].text for r in responses]

        finalized_entity_data = []
        for i, resp in enumerate(response_texts):
            datapoint = entity_data[message_ids2[i]]
            try:
                alt_names = response_validator(ListResponse, resp)['items']
                finalized_entity_data.append(
                    datapoint | {'entity_names': alt_names}
                )
            except:
                print("Error parsing entity names response.")
                finalized_entity_data.append(
                    datapoint | {'entity_names': []}
                )

        return finalized_entity_data
    

    def _page_locate(self):
        """
        Locates page numbers, for the identified measurements.

        Args:

        Returns:
            
        """
        instructions = PAGE_LOCATE_INSTRUCTIONS
        fields = list(self.identification_schema.model_fields.keys()) + ['measurement']
        messages = []
        message_tuples = []
        message_ids = []
        for i, datapoint in enumerate(self.data):
            item = {k: v for k,v in datapoint.items() if k in fields}
            entity_names = datapoint.get('entity_names', [])
            context = datapoint['context']
            pages = re.findall(r'<page number="(\d+)">', context)
            pages = [int(p) for p in pages]

            for p in pages:
                page_start = context.find(f'<page number="{p}">') + len(f'<page number="{p}">')
                page_end = context.find(f'</page>', page_start)
                page_text = context[page_start: page_end].strip()

                for m, measurement_names in datapoint['measurement_names'].items():
                    m_description = self.measurement_schema.model_fields[m].description

                    query = (
                        f"Does the given page contain a measurement "
                        f"for the feature {m_description} of the entity: {item}?"
                        f"Note that the entity may be referred to by any of the following aliases, extended names, or abbreviations: {entity_names}. "
                        f"Note that the measurement may be referred to by any of the following terms or abbreviations: {measurement_names}."
                    )
                    prompt = (
                        f"## Instructions:\n{instructions}\n\n## Context:\n{page_text}\n\n## Query:\n{query}"
                        )
                    messages.append(
                        [{"role": "user","content": prompt}]
                    )
                    message_tuples.append((instructions, context, query))
                    message_ids.append((i, p, m))

        if not self.probe:
            guided_decoding_params = GuidedDecodingParams(
                choice = ['true', 'false']
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
                # Isolate page in context
                if resp != 'false':
                    idx, page_number, measurement = message_ids[i]
                    datapoint = self.data[idx]
                    context = datapoint['context']
                    page_start = context.find(f'<page number="{page_number}">') + len(f'<page number="{page_number}">')
                    page_end = context.find(f'</page>', page_start)
                    page_text = context[page_start: page_end].strip()

                    page_id_data.append(
                        datapoint | 
                        {'page_number': page_number, 'context': page_text} |
                        {'measurement': measurement, 'measurement_id': i} | 
                        {'measurement_names': datapoint['measurement_names'].get(measurement, [])} |
                        {'page_logprob': response_probs[i]}
                    )

        else:
            responses = self.ctxlm.predict(
                prompts = message_tuples
            )

            page_id_data = []
            for i, resp in enumerate(responses):
                # Isolate page in context
                if resp['response'] != 'false':
                    idx, page_number, measurement = message_ids[i]
                    datapoint = self.data[idx]
                    context = datapoint['context']
                    page_start = context.find(f'<page number="{page_number}">') + len(f'<page number="{page_number}">')
                    page_end = context.find(f'</page>', page_start)
                    page_text = context[page_start: page_end].strip()

                    page_id_data.append(
                        datapoint | 
                        {'page_number': page_number, 'context': page_text} |
                        {'measurement': measurement, 'measurement_id': i} | 
                        {'measurement_names': datapoint['measurement_names'].get(measurement, [])} |
                        {'page_logprob': resp['logprob']} | {'page_attn_output': resp['attn_output']}
                    )

        return page_id_data
    

    '''
    def _in_table(self):
        """
        Locates page numbers, for the identified measurements.

        Args:

        Returns:
            
        """
        instructions = (
            f"You are an expert at finding data within research papers. "
            f"You will be given a passage from a research paper, and queried with a description for a specific entity to be measured. "
            f"Your task is to determine if the measurement occurs within a table or not. "
            f"Respond 'false' if the the given feature or entity do not appear in any table within the passage. "
            f"Respond 'false' if the given feature or entity do not appear in any table within the passage. "
            f"Respond 'false' if there is no table that explicity provides data for the given feature and entity. "
            f"Respond 'false' if there is no table that reports a direct numerical measurement for the feature and entity. "
            f"Respond 'false' if the data reported only contains values for parameter estimates or measures of fit for a statistical model. "
            f"Respond 'false' for cases where there is not a clear choice for a single, numerical data value. "
            f"Respond 'true' only if there is a table which explicity provides a direct numerical value measured for the given feature, with respect to the entity in question. "
            f"Respond with only 'true' or 'false', do not include any additional explanation in your response."
        )
        fields = list(self.identification_schema.model_fields.keys()) + ['measurement']
        messages = []
        message_tuples = []
        message_ids = []
        for i, datapoint in enumerate(self.data):
            item = {k: v for k,v in datapoint.items() if k in fields}
            page_number = datapoint.get('page_number')
            measurement = datapoint.get('measurement')
            measurement_description = self.measurement_schema.model_fields[measurement].description
            entity_names = datapoint.get('entity_names', [])
            measurement_names = datapoint.get('measurement_names', [])

            context = datapoint['context']
            #page_start = context.find(f'<page number="{page_number}">') + len(f'<page number="{page_number}">')
            #page_end = context.find(f'</page>', page_start)
            #page_text = context[page_start: page_end].strip()

            query = (
                f"Does a measurement of {measurement_description} for the entity {item} occur within a table in the passage?"
                f"Note that the entity may be referred to by any of the following extended names or abbreviations: {entity_names}. "
                f"Also note that the measurement may be referred to by any of the following abbreviations: {measurement_names}. "
            )
            prompt = (
                f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
                )
            messages.append(
                [{"role": "user","content": prompt}]
            )
            message_ids.append(i)


        guided_decoding_params = GuidedDecodingParams(
            choice = ['true', 'false']
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

        tabled_data = []
        for i, resp in enumerate(response_texts):
            datapoint = self.data[i]
            if resp == 'true':
                tabled_data.append(
                    datapoint | {'table_number': -2, 'table_logprob': 0.0}
                )
            else:
                tabled_data.append(
                    datapoint | {'table_number': -1, 'table_logprob': 0.0}
                )

        return tabled_data
    '''
    

    def _table_locate(self):
        """
        Locates page numbers, for the identified measurements.

        Args:

        Returns:
            
        """
        instructions = TABLE_LOCATE_INSTRUCTIONS
        fields = list(self.identification_schema.model_fields.keys()) + ['measurement']
        messages = []
        message_tuples = []
        message_ids = []
        for i, datapoint in enumerate(self.data):
            #if datapoint.get('table_number', -1) == -1:
            #    continue

            item = {k: v for k,v in datapoint.items() if k in fields}
            measurement = datapoint.get('measurement')
            measurement_description = self.measurement_schema.model_fields[measurement].description
            entity_names = datapoint.get('entity_names', [])
            measurement_names = datapoint.get('measurement_names', [])
            context = datapoint['context']
            tables = re.findall(r'<table number="(\d+)">', context)
            tables = [int(t) for t in tables]

            for t in tables:
                table_start = context.find(f'<table number="{t}">') + len(f'<table number="{t}">')
                table_end = context.find(f'</table>', table_start)
                table_text = context[table_start: table_end].strip()

                query = (
                    f"Does the table given as context contain a measurement "
                    f"for the feature {measurement_description} of the entity: {item}?"
                    f"Note that the entity may be referred to by any of the following aliases, extended names, or abbreviations: {entity_names}. "
                    f"Note that the measurement may be referred to by any of the following terms or abbreviations: {measurement_names}. "
                )
                prompt = (
                    f"## Instructions:\n{instructions}\n\n## Context:\n{table_text}\n\n## Query:\n{query}"
                    )
                messages.append(
                    [{"role": "user","content": prompt}]
                )
                message_tuples.append((instructions, table_text, query))
                message_ids.append((i, t))

        if not self.probe:
            guided_decoding_params = GuidedDecodingParams(
                choice = ['true', 'false']
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
                if resp != 'false':
                    idx, table_number = message_ids[i]
                    datapoint = self.data[idx]
                    table_id_data[idx] = datapoint | {
                        'table_number': table_number, 'table_logprob': response_probs[i]
                    }
            
        else:
            responses = self.ctxlm.predict(
                prompts = message_tuples
            )
            table_id_data = [
                d | {'table_number': -1, 'table_logprob': 0.0} for d in self.data
            ]
            for i, resp in enumerate(responses):
                if resp['response'] != 'false':
                    idx, table_number = message_ids[i]
                    datapoint = self.data[idx]

                    table_id_data[idx] = datapoint | {
                        'table_number': table_number,
                        'table_logprob': resp['logprob'],
                        'table_attn_output': resp['attn_output']
                    }

        return table_id_data
    

    def _measure_vllm(self):
        """
        Extracts measurements from the text chunks for the identified items.

        Args:

        Returns:
            
        """
        instructions = MEASURE_VALUE_INSTRUCTIONS
        messages = []
        message_ids = []
        for i, datapoint in enumerate(self.data):
            if datapoint['table_number'] == -1:
                context = datapoint['context']
                item = {k: v for k,v in datapoint.items() if k in self.identification_schema.model_fields.keys()}
                entity_names = datapoint.get('entity_names', [])
                measurement = datapoint['measurement']
                m_description = self.measurement_schema.model_fields[measurement].description
                measurement_names = datapoint.get('measurement_names', [])
                query = (
                    f"Extract the value for the {m_description} of the entity {item}. "
                    f"Note that the entity may be referred to by any of the following aliases, extended names, or abbreviations: {entity_names}. "
                    f"Note that the measurement may be referred to by any of the following terms or abbreviations: {measurement_names}. "
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

                # Get all td elements from the table (possible row names)
                #soup = BeautifulSoup(table_text, 'html.parser')
                #all_cells = soup.find_all(['td'])
                #cell_contents = [cell.get_text(strip=True) for cell in all_cells]

                item = {k: v for k,v in datapoint.items() if k in self.identification_schema.model_fields.keys()}
                entity_names = datapoint.get('entity_names', [])
                measurement = datapoint['measurement']
                measurement_names = datapoint.get('measurement_names', [])
                m_description = self.measurement_schema.model_fields[measurement].description
                query = (
                    f"Extract the row index name necessary to locate the measurement "
                    f"for the {m_description} of the entity {item} in table {table_number}. "
                    f"Note that the entity may be referred to by any of the following aliases, extended names, or abbreviations: {entity_names}. "
                    f"Note that the measurement may be referred to by any of the following terms or abbreviations: {measurement_names}. "
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

                item = {k: v for k,v in datapoint.items() if k in self.identification_schema.model_fields.keys()}
                entity_names = datapoint.get('entity_names', [])
                measurement = datapoint['measurement']
                m_description = self.measurement_schema.model_fields[measurement].description
                measurement_names = datapoint.get('measurement_names', [])
                query = (
                    f"Extract the column index name necessary to locate the measurement "
                    f"for the {m_description} of the entity {item} in table {table_number}. "
                    f"Note that the entity may be referred to by any of the following aliases, extended names, or abbreviations: {entity_names}. "
                    f"Note that the measurement may be referred to by any of the following terms or abbreviations: {measurement_names}. "
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

                '''
                # Find column index:
                if col_name in table.columns:
                    col_idx = table.columns.get_loc(col_name)
                else:
                    try:
                        # parse multi-level column names
                        col_name_parsed = ast.literal_eval(col_name)
                        col_idx = table.columns.get_loc(col_name_parsed)
                    except:
                        col_idx = None


                # Find row index:
                try:
                    row_mask = (table.iloc[:,:3] == row_name).sum(axis = 1)
                    row_idx = row_mask[row_mask > 0].index[0]
                except:
                    row_idx = None

                if row_idx is not None and col_idx is not None:
                    table_extracted_data[i] = datapoint | {
                        'value': table.iat[row_idx, col_idx]
                    }
                '''

        return [t for t in table_extracted_data if t.get('value', None) is not None]
    

    def _standardize_measurements(self):
        """
        Standardizes the measurement units for the extracted measurements.

        Args:

        Returns:
            
        """
        instructions = STANDARDIZE_MEASUREMENTS_INSTRUCTIONS
        messages = []
        message_data_ids = []
        sampling_params = []
        for i, datapoint in enumerate(self.data):
            item = {k: v for k,v in datapoint.items() if k in self.identification_schema.model_fields.keys()}
            entity_names = datapoint.get('entity_names', [])
            measurement = datapoint['measurement']
            measurement_val = datapoint['value']
            measurement_description = self.measurement_schema.model_fields[measurement].description
            measurement_names = datapoint.get('measurement_names', [])
            context = datapoint['context']

            query = (
                f"Entity measured: {item}\n"
                f"Measurement type: {measurement_description}\n"
                f"Measurement value: {measurement_val}\n"
                f"Standardize the measurement value for the given data point. "
                f"Note that the entity may be referred to by any of the following extended aliases, names, or abbreviations: {entity_names}. "
                f"Note that the measurement may be referred to by any of the following terms or abbreviations: {measurement_names}. "
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
        
    
    def _standardize_units(self):
        """
        Gives standardized units to the extracted measurements.

        Args:

        Returns:
            
        """
        instructions = STANDARDIZE_UNITS_INSTRUCTIONS
        messages = []
        message_data_ids = []
        sampling_params = []
        for i, datapoint in enumerate(self.data):
            item = {k: v for k,v in datapoint.items() if k in self.identification_schema.model_fields.keys()}
            entity_names = datapoint.get('entity_names', [])
            measurement = datapoint['measurement']
            measurement_val = datapoint['value']
            measurement_description = self.measurement_schema.model_fields[measurement].description
            measurement_names = datapoint.get('measurement_names', [])
            available_units = self.measurement_schema.model_fields[measurement].json_schema_extra.get('units', None)

            if available_units is not None:
                units_list = available_units #+ ['other']
                context = datapoint['context']
                query = (
                    f"Entity measured: {item}\n"
                    f"Measurement type: {measurement_description}\n"
                    f"Measurement value: {measurement_val}\n"
                    f"Determine the unit of measurement for the given data point from among the following choices: {available_units}. "
                    f"Note that the entity may be referred to by any of the following aliases, extended names, or abbreviations: {entity_names}. "
                    f"Note that the measurement may be referred to by any of the following terms or abbreviations: {measurement_names}. "
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
        self.data = self._locate()
        self.data = self._measure_vllm()
        self.data = self._measure_vllm_tables()
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



