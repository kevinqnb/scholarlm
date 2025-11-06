# ScholarLM :microscope: :books:

**Parse and analyze scientific research papers with large language models using mechanistic interpretability.**

*Please note:* This project is a work in progress. 

This library implements a system for extracting insights from scientific papers (which are in the form of pdfs) using large language models.
Specifically, we apply local and open source LLMs towards organized tasks for:
* Document OCR: translating pdf images into markdown, and splitting into paragraph sized chunks.
* Document extraction: systematically collecting data points from chunks of markdown text. 
* Hallucination detection: mechanistic intervention on model activations to detect and prevent hallucinated responses. 

Our focus is on using small, local models for OCR and text generation tasks, and this library is designed to be compatible 
with any such model of your choosing. 

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/scholarlm.git
cd scholarlm

# Install with pixi (recommended)
pixi install

# Or install with pip
pip install -e .
```

### Basic Usage

```python
from scholarlm import ContextLM

# Initialize the model
model = ContextLM(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    top_k=0.1,  # Top 10% of context tokens to analyze
    max_new_tokens=50
)

# Generate text with context analysis
context = "The Earth orbits around the Sun in an elliptical path."
instructions = "Explain planetary motion."

result = model.generate(context, instructions)

print(f"Response: {result['response']}")
print(f"Parametric Score: {result['parametric_score']:.4f}")
print(f"Context Score: {result['context_score']:.4f}")
```

## References

ScholarLM implements external context and parametric knowledge score methods from:
> Sun, Zhongxiang, et al. "ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability." ICLR. 2025.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
