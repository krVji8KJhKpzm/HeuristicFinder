# GLM (General Language Model) API Integration for rl4co

This document describes the GLM API integration for the rl4co heuristic finder, which allows you to use GLM models for generating potential functions in combinatorial optimization problems.

## Overview

The GLM integration adds support for GLM (General Language Model) APIs to the existing rl4co LLM framework. GLM models, developed by Zhipu AI, are powerful language models that can be used for code generation tasks including potential function design for PBRS (Potential-Based Reward Shaping).

## Features

- **OpenAI-compatible API**: Uses the same interface as DeepSeek and Kimi APIs
- **Multiple GLM Models**: Support for various GLM models including glm-4, glm-3-turbo, etc.
- **Environment Variable Configuration**: Easy configuration through environment variables
- **Unified Interface**: Seamlessly integrates with existing `generate_candidates()` function
- **Two-stage Generation**: Supports both single-stage and two-stage generation patterns

## Setup

### 1. Get GLM API Key

First, you need to obtain a GLM API key from [Zhipu AI Platform](https://open.bigmodel.cn/).

### 2. Set Environment Variables

```bash
export GLM_API_KEY="your-glm-api-key"
```

Optional configuration:
```bash
export GLM_API_BASE="https://open.bigmodel.cn/api/paas/v4"  # Default
export GLM_MODEL="glm-4"  # Default model
export GLM_TEMPERATURE="0.0"  # Default temperature
export GLM_MAX_TOKENS="1024"  # Default max tokens
```

## Usage

### Basic Usage

```python
from rl4co.heuristic_finder.llm import generate_candidates_via_glm, format_prompt

# Create a prompt
prompt = format_prompt(env_name="tsp", guidance="Design a simple potential function")

# Generate candidates using GLM
candidates = generate_candidates_via_glm(prompt=prompt, n=1, debug=True)

# Use the generated code
if candidates:
    print("Generated potential function:")
    print(candidates[0])
```

### Using Different Models

```python
# Use glm-3-turbo for faster generation
candidates = generate_candidates_via_glm(
    prompt=prompt,
    model="glm-3-turbo",
    n=1
)

# Use glm-4 for higher quality
candidates = generate_candidates_via_glm(
    prompt=prompt,
    model="glm-4",
    n=1
)
```

### Unified Interface

You can also use GLM through the unified interface:

```python
import os
from rl4co.heuristic_finder.llm import generate_candidates

# Set GLM as the preferred provider
os.environ['LLM_API_PROVIDER'] = 'glm'

# Use the unified interface
candidates = generate_candidates(prompt=prompt, n=1)
```

### Two-stage Generation

```python
from rl4co.heuristic_finder.llm import two_stage_generate_candidates

# Use GLM for two-stage generation
candidates = two_stage_generate_candidates(
    prompt=prompt,
    n=1,
    env_name="tsp",
    api_provider="glm",
    debug=True
)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GLM_API_KEY` | Your GLM API key (required) | None |
| `GLM_API_BASE` | GLM API base URL | `https://open.bigmodel.cn/api/paas/v4` |
| `GLM_MODEL` | Default model name | `glm-4` |
| `GLM_REASONER_MODEL` | Model for reasoning stage | `glm-4` |
| `GLM_TEMPERATURE` | Temperature for generation | `0.0` |
| `GLM_MAX_TOKENS` | Maximum tokens in response | `1024` |

## Available GLM Models

- **glm-4**: Latest and most capable model (recommended)
- **glm-3-turbo**: Faster, more cost-effective model
- **glm-3-130b**: Larger model for complex tasks
- **glm-2**: Earlier generation model

## Provider Priority

The provider selection follows this priority order:

1. **Explicit preference**: `api_provider="glm"` parameter
2. **Environment variable**: `LLM_API_PROVIDER=glm`
3. **API key availability**: Checks for `GLM_API_KEY`
4. **Fallback**: Defaults to DeepSeek if no GLM API key is found

## Error Handling

The GLM integration includes comprehensive error handling:

```python
candidates = generate_candidates_via_glm(prompt=prompt, debug=True)

if not candidates:
    print("No candidates generated. Check:")
    print("- GLM_API_KEY is set correctly")
    print("- Network connectivity")
    print("- API rate limits")
    print("- Model availability")
```

## Testing

Run the provided test script to verify your GLM integration:

```bash
# Set your API key
export GLM_API_KEY="your-api-key"

# Run the test
python test_glm_integration.py
```

## Integration with Existing Code

The GLM integration works seamlessly with existing rl4co code:

```python
# In your existing heuristic finder code
from rl4co.heuristic_finder.llm import eoh_llm_i1, eoh_llm_e1, eoh_llm_m1

# Set GLM as provider
os.environ['LLM_API_PROVIDER'] = 'glm'

# Use existing functions - they will automatically use GLM
candidates = eoh_llm_i1(model="glm-4", n=1, env_name="tsp")
```

## Performance Considerations

- **Model Selection**: Use `glm-3-turbo` for faster generation, `glm-4` for higher quality
- **Temperature**: Lower values (0.0-0.3) provide more consistent results
- **Batch Size**: Consider rate limits when generating multiple candidates
- **Network**: Ensure stable network connection to the GLM API servers

## Troubleshooting

### Common Issues

1. **No candidates generated**
   - Verify `GLM_API_KEY` is set correctly
   - Check network connectivity
   - Ensure your API key has sufficient credits

2. **API timeouts**
   - Increase timeout values if needed
   - Try a different model (glm-3-turbo is faster)
   - Check GLM API status

3. **Poor code quality**
   - Try different temperature settings
   - Use a more capable model (glm-4)
   - Refine your prompt with more specific guidance

### Debug Mode

Enable debug mode to see detailed information:

```python
candidates = generate_candidates_via_glm(
    prompt=prompt,
    debug=True,  # Enable debug output
    n=1
)
```

## Comparison with Other Providers

| Feature | GLM | DeepSeek | Kimi |
|---------|-----|----------|------|
| API Compatibility | OpenAI-compatible | OpenAI-compatible | OpenAI-compatible |
| Default Model | glm-4 | deepseek-chat | kimi-k2-thinking |
| Base URL | open.bigmodel.cn | api.deepseek.com | api.kimi.ai |
| Strengths | Chinese language, code generation | Reasoning, math | Long context, reasoning |

## Future Enhancements

- Support for GLM-4V (vision capabilities)
- Fine-tuned models for specific optimization problems
- Advanced prompting techniques for better code generation
- Integration with GLM's function calling capabilities

## Support

For issues related to:
- **GLM API**: Contact Zhipu AI support or check their documentation
- **rl4co integration**: Create an issue in the rl4co repository
- **Test script**: Run `python test_glm_integration.py` for diagnostics

## License

This integration follows the same license as the main rl4co project.
