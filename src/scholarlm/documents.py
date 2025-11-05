from langchain_text_splitters import RecursiveCharacterTextSplitter

class MarkdownDocuments:
    """
    A class to manage and process markdown documents for use with a language model.

    Args:
        chunk_size (int): The maximum token size of each chunk.
        overlap (int): The number of overlapping characters between consecutive chunks.
    Attributes:
        document_info (list[dict[str, dict]]): A list to store metadata about each document. Each 
            item in the list is a dictionary with a single key-value pair, where the key is 
            the document's filename and the value is another dictionary containing key-value
            metadata, e.g. {'author': 'et al', 'title': 'xyz', 'date': '2028'}.
        documents (list): A list to store the actual document contents.
        chunks (list): A list to store processed chunks of documents.
        chunk_labels (list): A list to store labels or identifiers for each chunk. Specifically, 
            each label is an index referring to the document from which the chunk was derived.
    """
    def __init__(
            self,
            chunk_size: int = 512,
            overlap: int = 0,
            separators: list[str] = None
        ):
        self.chunk_size = chunk_size
        self.overlap = overlap

        if separators is None:
            separators = [
                "<TABLE>", "</TABLE>", "<SECTION>", "</SECTION>", "<PARAGRAPH>", "</PARAGRAPH>"
            ]
        self.separators = separators

        self.document_info = []
        self.documents = []
        self.chunks = []
        self.chunk_labels = []


    def load(self, document_info : list[dict[str, dict]]):
        """
        Load documents based on provided metadata.

        Args:
            document_info (list[dict[str, dict]]): A list to store metadata about each document. 
                Each item in the list is a dictionary with a single key-value pair, where the key is 
                the document's filename and the value is another dictionary containing key-value
                metadata, e.g. {'author': 'et al', 'title': 'xyz', 'date': '2028'}.
        """
        self.document_info += document_info
        for doc_meta in document_info:
            for filename in doc_meta:
                with open(filename, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.documents.append(content)


    def split(self):
        """
        Split loaded documents into smaller chunks, using a token-based approach.

        Args:
            chunk_size (int): The maximum token size of each chunk.
            overlap (int): The number of overlapping characters between consecutive chunks.
        """
        for doc_index, document in enumerate(self.documents):
            # Run the recursive splitter
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                separators = self.separators
            )
            doc_chunks = text_splitter.split_text(document)

            # Remove separators from the text
            for i, sep in enumerate(self.separators):
                doc_chunks = [chunk.replace(sep, "") for chunk in doc_chunks]

            doc_chunks = [t.strip() for t in doc_chunks if len(t.strip()) > 0]
            self.chunks += [doc_chunks]
            #self.chunk_labels += [doc_index] * len(doc_chunks)

        return self.chunks #, self.chunk_labels


