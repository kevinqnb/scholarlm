from pydantic import BaseModel
import pandas as pd
import re
from io import StringIO
from bs4 import BeautifulSoup
import torch
from enum import Enum
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import ast

from .contextlm2 import ContextLM2
from scholarlm.utils import table_extract


def response_validator(response_structure, response):
    pyd = response_structure.model_validate_json(response)
    out_dict = pyd.model_dump()
    return out_dict

class BooleanResponse(BaseModel):
    answer: bool

class IntegerResponse(BaseModel):
    value: int

class ListResponse(BaseModel):
    items: list[str]

class LocationResponse(BaseModel):
    page_number: int
    table_number: int

def create_multichoice_schema(choices: list[str]):
    # Dynamically create an Enum from your choices
    ChoiceEnum = Enum('ChoiceEnum', {choice: choice for choice in choices}, type=str)
    
    class MultiChoice(BaseModel):
        selections: list[ChoiceEnum]
    
    return MultiChoice

class TableIdResponse(BaseModel):
    row_indices: list[str] | None
    column_indices: list[str] | None


class MeasurementLM2:
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
        page_selection : bool = False,
        use_llm : bool = True,
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
        self.page_selection = page_selection

        if use_llm:
            self.llm = LLM(model=model_name)
        else:
            self.llm = None

        if self.return_full_output:
            ctxlm_params = {k: v for k,v in sampling_params.items() if k not in ['max_tokens', 'seed', 'temperature', 'stop']}
            ctxlm_params['do_sample'] = False
            ctxlm_params['max_new_tokens'] = 20
            self.ctxlm = ContextLM2(
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                #model_name = "meta-llama/Llama-3.1-8B",                            
                #model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
                top_k = 20,
                sampling_params=ctxlm_params,
                nnsight_kwargs = {"torch_dtype": torch.bfloat16},
                #nnsight_kwargs = {"load_in_8bit": True},
                cache_dir = self.cache_dir
            )
        

    
    def _identify_measurements(self):
        """
        Identifies measurements in the text based on the measurement schema.

        Args:
            
        Returns:
            
        """
        instructions = (
            f"You are an expert at identifying relevant information in scientific texts. "
            f"Determine if the given context contains any information relevant to the requested query. "
            f"Respond 'true' if the context is relevant and 'false' if it is not. "
        ) 
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
                    {"role": "user", "content": prompt}]
                )
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

        instructions2 = (
            f"You are an expert in understanding scientific texts. Given the context and a specific measurement type, "
            f"extract any abbreviations used to directly refer to the measurement type in question. "
            f"Do not include any abbreviations that refer to similar concepts or measurements, but which are not direct in relation. "
            f"Structure your response as a list of strings, for example: ['abbreviation_1', 'abbreviation_2']"
            f"If there are no abbreviations, respond with an empty list: []"
        )
        messages2 = []
        message_ids2 = []
        for i, resp in enumerate(response_texts):
            idx, measurement = message_ids[i]
            if resp == 'true':
                datapoint = self.data[idx]
                context = datapoint['context']
                measurement_description = self.measurement_schema.model_fields[measurement].description
                query = (
                    f"Extract any abbreviations that refer to "
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


        instructions2 = (
            f"You are an expert in understanding scientific texts. Given the context and a specific entity, "
            f"extract any abbreviations or extended names used to directly refer to the entity in question. "
            f"Do not include any abbreviations or names that refer to similar entities or concepts, but which are not direct and specific in relation. "
            f"Structure your response as a list of strings, for example: ['abbreviation_1', 'name_2']"
            f"If there are no abbreviations or extended names, respond with an empty list: []"
        )
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
        instructions = (
            f"You are an expert at finding data within research papers. "
            f"You will be queried with a description of a specific entity to be measured, "
            f"and asked to determine if the measurement occurs on a single, given page of context. "
            f"Respond 'false' if the the given feature or entity do not appear in the context. "
            f"Respond 'false' if the context does not explicity provide data for the given feature and entity. "
            f"Respond 'false' if the data reported is not either a direct numerical measurement or a mean of numerical measurements. "
            f"Respond 'false' if the data reported only contains values for parameter estimates or other statistical measures of fit. "
            f"Respond 'false' for ranges of values, inequalties, or other cases where there is not a clear choice for a single numerical data value. "
            f"Respond 'true' only if the context explicity provides a direct numerical value measured for the given feature, with respect to the entity in question. "
        )
        fields = list(self.identification_schema.model_fields.keys()) + ['measurement']
        messages = []
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
                        f"of {m_description} for the following entity: {item}?"
                        #f"Note that the entity may be referred to by any of the following extended names or abbreviations: {entity_names}. "
                        f"Note that the measurement may be referred to by any of the following abbreviations: {measurement_names}. "
                    )
                    prompt = (
                        f"## Instructions:\n{instructions}\n\n## Context:\n{page_text}\n\n## Query:\n{query}"
                        )
                    messages.append(
                        [{"role": "user","content": prompt}]
                    )
                    message_ids.append((i, p, m))

        guided_decoding_params = GuidedDecodingParams(
            choice = ['true', 'false']
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )

        responses = self.llm.chat(
            messages = messages,
            sampling_params = sampling_params,
        )
        response_texts = [r.outputs[0].text for r in responses]
        page_id_data = []
        for i, resp in enumerate(response_texts):
            # Isolate page in context
            if resp != 'false':
                idx, page_number, measurement = message_ids[i]
                datapoint = self.data[idx]
                '''
                if self.page_selection:
                    # Slice context to include only the relevant information
                    context = ''
                    page_tag_start = f'<page number="{page_number}">'
                    page_tag_end = f'</page>'
                    context_start_idx = self.data[idx]['context'].find(page_tag_start) + len(page_tag_start)
                    context_end_idx = self.data[idx]['context'].find(page_tag_end)
                    context = self.data[idx]['context'][context_start_idx: context_end_idx].strip()

                    page_id_data.append(
                        self.data[idx] | {'page_number': page_number} |
                        {'context': context}
                    )
                else:
                    page_id_data.append(
                        self.data[idx] | {'page_number': page_number}
                    )
                '''
                page_id_data.append(
                    datapoint | 
                    {'page_number': page_number, 'measurement': measurement, 'measurement_id': i} | 
                    {'measurement_names': datapoint['measurement_names'].get(measurement, [])}
                )

        return page_id_data


    def _table_locate(self):
        """
        Locates table numbers, for the identified measurements.

        Args:

        Returns:
            
        """
        instructions = (
            f"You are an expert at finding data within research papers. "
            f"You will be queried with a description of an specific entity to be measured, "
            f"and asked to locate and identify the table number in which the measurement appears. "
            f'A measurement appears in table j if it can be found within the <table number="j">...</table> tags in the context. '
            f"To find the table number, look for the last numbered table tag that appears before the measurement in the context. "
            f"If the measurement appears in the text but not within a table, respond with the value -1 for the table number. "
            f"Do not attempt to infer or guess table numbers. "
            f"Only use table tags to locate measurements, and do not use table or page numbers mentioned in figure captions, footnotes, or references. "
            f"Respond only with the table number and do not include any additional text or explanation. "
        )
        fields = list(self.identification_schema.model_fields.keys()) + ['measurement']
        messages = []
        for i, datapoint in enumerate(self.data):
            item = {k: v for k,v in datapoint.items() if k in fields}
            entity_names = datapoint.get('entity_names', [])
            m_description = self.measurement_schema.model_fields[datapoint['measurement']].description
            measurement_names = datapoint.get('measurement_names', [])
            context = datapoint['context']
            page_number = datapoint.get('page_number')
            page_start = context.find(f'<page number="{page_number}">') + len(f'<page number="{page_number}">')
            page_end = context.find(f'</page>', page_start)
            page_text = context[page_start: page_end].strip()

            query = (
                f"Identify the table number which contains the measurement "
                f"of {m_description} ({datapoint['measurement']}) for the following entity: {item}."
                #f"Note that the entity may be referred to by any of the following extended names or abbreviations: {entity_names}. "
                f"Note that the measurement may be referred to by any of the following abbreviations: {measurement_names}. "
            )
            prompt = (
                f"## Instructions:\n{instructions}\n\n## Context:\n{page_text}\n\n## Query:\n{query}"
                )
            messages.append(
                [{"role": "user","content": prompt}]
            )

        guided_decoding_params = GuidedDecodingParams(
            json=IntegerResponse.model_json_schema()
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )

        responses = self.llm.chat(
            messages = messages,
            sampling_params = sampling_params,
        )
        response_texts = [r.outputs[0].text for r in responses]
        page_id_data = []
        for i, resp in enumerate(response_texts):
            try:
                resp_validated = response_validator(IntegerResponse, resp)
                table_number = resp_validated['value']
            except:
                print("Validation error in integer response.")
                table_number = -1
            
            # Isolate page in context
            if table_number != -1:
                if self.page_selection:
                    # Slice context to include only the relevant information
                    context = ''
                    table_tag_start = f'<table number="{table_number}">'
                    table_tag_end = f'</table>'
                    context_start_idx = self.data[i]['context'].find(table_tag_start) + len(table_tag_start)
                    context_end_idx = self.data[i]['context'].find(table_tag_end)
                    context = self.data[i]['context'][context_start_idx: context_end_idx].strip()

                    page_id_data.append(
                        self.data[i] | {'table_number': table_number} |
                        {'context': context}
                    )

                else:
                    page_id_data.append(
                        self.data[i] | {'table_number': table_number}
                    )

        return page_id_data
    

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
            f"Respond with the value None if the the given feature or entity do not appear in the context. "
            f"Respond with the value None if the context does not explicity provide data for the given feature and entity. "
            f"Respond with the value None if the data reported is not either a direct numerical measurement or a mean of numerical measurements. "
            f"Respond with the value None if the data reported only contains values for parameter estimates or other statistical measures of fit. "
            f"Respond with the value None for ranges of values, inequalties, or other cases where there is not a clear choice for a single numerical data value. "
            f"Respond with the extracted value only if the context explicity provides a direct numerical value measured for the given feature, with respect to the entity in question. "
            f"Copy the value exactly as it appears in the context. "
            f"Give the value only, and do not include any units of measurement, descriptors, or explanation in your response. "
            f"If the value is associated with uncertainty measures (e.g., ± values, confidence intervals), report only the central value without any uncertainty information. "
        )
        messages = []
        message_ids = []
        for i, datapoint in enumerate(self.data):
            if datapoint['table_number'] == -1:
                # Zoom in on the specific text:
                context = datapoint['context']
                page_number = datapoint['page_number']
                page_start = context.find(f'<page number="{page_number}">') + len(f'<page number="{page_number}">')
                page_end = context.find(f'</page>', page_start)
                page_text = context[page_start: page_end].strip()

                item = {k: v for k,v in datapoint.items() if k in self.identification_schema.model_fields.keys()}
                entity_names = datapoint.get('entity_names', [])
                measurement = datapoint['measurement']
                m_description = self.measurement_schema.model_fields[measurement].description
                measurement_names = datapoint.get('measurement_names', [])
                query = (
                    f"Extract the value for the {m_description} of the entity {item}. "
                    #f"Note that the entity may be referred to by any of the following extended names or abbreviations: {entity_names}. "
                    f"Note that the measurement may be referred to by any of the following abbreviations: {measurement_names}. "
                )
                prompt = (
                    f"## Instructions:\n{instructions}\n\n## Context:\n{page_text}\n\n## Query:\n{query}"
                    )
                messages.append([
                    {"role": "user","content": prompt}]
                )
                message_ids.append(i)

        #mlm_params = {k: v for k,v in self.sampling_params.items() if k not in ['max_tokens', 'seed', 'temperature', 'stop']}
        #mlm_params['temperature'] = 0.0
        #mlm_params['max_tokens'] = 20
        #sampling_params = SamplingParams(
        #    **mlm_params
        #)
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
        instructions = (
            f"You are an expert in extracting precise numerical data from user provided, scientific text. "
            f"You will be given an html representation of a table from a research paper, "
            f"and queried with a description of a specific entity to be measured, and a specific measurement type to report on. "
            f"Your task is to extract the row index name necessary to locate the measurement within the table. "
            f"If there are multiple row names that could apply, choose the one that is most specific to the entity or measurement in question. "
            f"Your response must use the exact row index name as it appears in the table. "
            f"If there is no row name corresponding to the relevant entity or measurement, respond 'None'. "
            f"Otherwise, respond with the row name only, without any additional text or explanation."
        )

        messages = []
        sampling_params = []
        message_ids = []
        for i, datapoint in enumerate(self.data):
            if datapoint['table_number'] != -1:
                # Zoom in on the specific table:
                context = datapoint['context']
                table_number = datapoint['table_number']
                table_start = f'<table number="{table_number}">'
                context_start_idx = context.find(table_start)
                if context_start_idx == -1:
                    continue

                table_context_start = context[context_start_idx:]
                table_tag_end = f'</table>'
                context_end_idx = table_context_start.find(table_tag_end) + len(table_tag_end)
                table_text = table_context_start[:context_end_idx].strip()

                # Get all td elements from the table (possible row names)
                soup = BeautifulSoup(table_text, 'html.parser')
                all_cells = soup.find_all(['td'])
                cell_contents = [cell.get_text(strip=True) for cell in all_cells]

                item = {k: v for k,v in datapoint.items() if k in self.identification_schema.model_fields.keys()}
                entity_names = datapoint.get('entity_names', [])
                measurement = datapoint['measurement']
                measurement_names = datapoint.get('measurement_names', [])
                m_description = self.measurement_schema.model_fields[measurement].description
                query = (
                    f"Extract the row index name necessary to locate the measurement "
                    f"for the {m_description} of the entity {item} in table {table_number}. "
                    #f"Note that the entity may be referred to by any of the following extended names or abbreviations: {entity_names}. "
                    f"Note that the measurement may be referred to by any of the following abbreviations: {measurement_names}. "
                )
                prompt = (
                    f"## Instructions:\n{instructions}\n\n## Context:\n{table_text}\n\n## Query:\n{query}"
                )
                messages.append([
                    {"role": "user","content": prompt}]
                )
                guided_decoding_params = GuidedDecodingParams(
                    choice = cell_contents + ['None']
                )
                params = SamplingParams(
                    **self.sampling_params,
                    guided_decoding=guided_decoding_params
                )
                sampling_params.append(params)
                message_ids.append(i)

        #sampling_params = SamplingParams(
        #    **self.sampling_params
        #)

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
        instructions = (
            f"You are an expert in extracting precise numerical data from user provided, scientific text. "
            f"You will be given an html representation of a table from a research paper, "
            f"and queried with a description of a specific entity to be measured, and a specific measurement type to report on. "
            f"Your task is to extract the column index name necessary to locate the measurement within the table. "
            f"If there are multiple column names that could apply, choose the one that is most specific to the entity or measurement in question. "
            f"Your response must use the exact column index name as it appears in the table. "
            f"If there is no column name corresponding to the relevant entity or measurement, respond 'None'. "
            f"Otherwise, respond with the column name only, without any additional text or explanation."
        )

        messages = []
        sampling_params = []
        message_ids = []
        for i, datapoint in enumerate(self.data):
            if datapoint['table_number'] != -1:
                # Zoom in on the specific table:
                context = datapoint['context']
                table_number = int(datapoint['table_number'])
                table_start = f'<table number="{table_number}">'
                context_start_idx = context.find(table_start)
                if context_start_idx == -1:
                    continue

                table_context_start = context[context_start_idx:]
                table_tag_end = f'</table>'
                context_end_idx = table_context_start.find(table_tag_end) + len(table_tag_end)
                table_text = table_context_start[:context_end_idx].strip()

                tables = pd.read_html(StringIO(datapoint['context']))
                table_df = tables[table_number]
                column_names = [str(name) for name in table_df.columns.tolist()]

                item = {k: v for k,v in datapoint.items() if k in self.identification_schema.model_fields.keys()}
                entity_names = datapoint.get('entity_names', [])
                measurement = datapoint['measurement']
                m_description = self.measurement_schema.model_fields[measurement].description
                measurement_names = datapoint.get('measurement_names', [])
                query = (
                    f"Extract the column index name necessary to locate the measurement "
                    f"for the {m_description} of the entity {item} in table {table_number}. "
                    #f"Note that the entity may be referred to by any of the following extended names or abbreviations: {entity_names}. "
                    f"Note that the measurement may be referred to by any of the following abbreviations: {measurement_names}. "
                )
                prompt = (
                    f"## Instructions:\n{instructions}\n\n## Context:\n{table_text}\n\n## Query:\n{query}"
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

    '''
    def _measure_vllm_columns(self):
        """
        Extracts measurements from tables in the text.

        Args:

        Returns:
            
        """
        instructions = (
            f"You are an expert in extracting precise numerical data from user provided, scientific text. "
            f"You will be given a list of column names from a table in a research paper, "
            f"and queried with a description of a specific entity to be measured, and a specific measurement type to report on. "
            f"Your task is to choose the column index name which best fits the requested entity or measurement. "
            f"If there is no column name corresponding to the relevant entity or measurement, respond 'None'. "
            f"Otherwise, respond with the column name only, without any additional text or explanation."
        )
        messages = []
        sampling_params = []
        message_ids = []
        for i, datapoint in enumerate(self.data):
            if datapoint['table_number'] != -1:
                # Zoom in on the specific table:
                table_number = int(datapoint['table_number'])
                tables = pd.read_html(StringIO(datapoint['context']))
                table = tables[table_number]
                column_names = [str(name) for name in table.columns.tolist()]

                item = {k: v for k,v in datapoint.items() if k in self.identification_schema.model_fields.keys()}
                measurement = datapoint['measurement']
                m_description = self.measurement_schema.model_fields[measurement].description
                query = "Extract the column index name necessary to locate the measurement for the " + f"{m_description} of the entity {item}."
                prompt = (
                    f"## Instructions:\n{instructions}\n\n## Context:\n{column_names}\n\n## Query:\n{query}"
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
    '''


    def _table_extract(self):
        """
        Extracts measurements from tables in the text.
        """
        table_extracted_data = [d for d in self.data]
        for i, datapoint in enumerate(self.data):
            if datapoint.get('row_index', None) is not None and datapoint.get('column_index', None) is not None:
                table_number = int(datapoint['table_number'])
                tables = pd.read_html(StringIO(datapoint['context']))
                table = tables[table_number]

                col_name = datapoint['column_index']
                row_name = datapoint['row_index']

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
                    row_mask = (table == row_name).sum(axis = 1)
                    row_idx = row_mask[row_mask > 0].index[0]
                except:
                    row_idx = None

                if row_idx is not None and col_idx is not None:
                    table_extracted_data[i] = datapoint | {
                        'value': table.iat[row_idx, col_idx]
                    }

        return [t for t in table_extracted_data if t.get('value', None) is not None]
    

    def _standardize_measurements(self):
        """
        Standardizes the measurement units for the extracted measurements.

        Args:

        Returns:
            
        """
        instructions = (
            f"You are an expert in data collection and scientific measurements. "
            f"You will be given context from a research paper, along with a description of a extracted measurement value and the feature and entity it was reported for. "
            f"Your task is to standardize the measurement values, and format them in a manner which is easier to parse for downstream tasks. "
            f"For values associated with uncertainty measures (e.g., ± values, confidence intervals), report only the central value without any uncertainty information. "
            f"For values which are reported with a unit of measurement or other descriptor, convert the value to a standardized numerical format without any units or descriptors. "
            f"Your response should include the standardized value only, do not include any additional explanation or text. "

        )        
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
            page_number = datapoint.get('page_number')
            page_start = context.find(f'<page number="{page_number}">') + len(f'<page number="{page_number}">')
            page_end = context.find(f'</page>', page_start)
            page_text = context[page_start: page_end].strip()
            query = (
                f"Entity measured: {item}\n"
                f"Measurement type: {measurement_description}\n"
                f"Measurement value: {measurement_val}\n"
                f"Standardize the measurement value for the given data point. "
                #f"Note that the entity may be referred to by any of the following extended names or abbreviations: {entity_names}. "
                f"Note that the measurement may be referred to by any of the following abbreviations: {measurement_names}. "
            )
            prompt = (
                f"## Instructions:\n{instructions}\n\n## Context:\n{page_text}\n\n## Query:\n{query}"
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
        instructions = (
            f"You are an expert in data collection and scientific measurements. "
            f"You will be given context from a research paper, along with a description of a extracted measurement value and the feature and entity it was reported for. "
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
                    #f"Note that the entity may be referred to by any of the following extended names or abbreviations: {entity_names}. "
                    f"Note that the measurement may be referred to by any of the following abbreviations: {measurement_names}. "
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



