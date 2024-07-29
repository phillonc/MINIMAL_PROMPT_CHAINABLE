import random
from chain import FusionChain, FusionChainResult, MinimalChainable


def test_chainable_solo():
    # Mock model and callable function
    class MockModel:
        pass

    def mock_callable_prompt(model, prompt):
        return f"Solo response: {prompt}"

    # Test context and single chain
    context = {"variable": "Test"}
    chains = ["Single prompt: {{variable}}"]

    # Run the Chainable
    result, _ = MinimalChainable.run(context, MockModel(), mock_callable_prompt, chains)

    # Assert the results
    assert len(result) == 1
    assert result[0] == "Solo response: Single prompt: Test"


def test_chainable_run():
    # Mock model and callable function
    class MockModel:
        pass

    def mock_callable_prompt(model, prompt):
        return f"Response to: {prompt}"

    # Test context and chains
    context = {"var1": "Hello", "var2": "World"}
    chains = ["First prompt: {{var1}}", "Second prompt: {{var2}} and {{var1}}"]

    # Run the Chainable
    result, _ = MinimalChainable.run(context, MockModel(), mock_callable_prompt, chains)

    # Assert the results
    assert len(result) == 2
    assert result[0] == "Response to: First prompt: Hello"
    assert result[1] == "Response to: Second prompt: World and Hello"


def test_chainable_with_output():
    # Mock model and callable function
    class MockModel:
        pass

    def mock_callable_prompt(model, prompt):
        return f"Response to: {prompt}"

    # Test context and chains
    context = {"var1": "Hello", "var2": "World"}
    chains = ["First prompt: {{var1}}", "Second prompt: {{var2}} and {{output[-1]}}"]

    # Run the Chainable
    result, _ = MinimalChainable.run(context, MockModel(), mock_callable_prompt, chains)

    # Assert the results
    assert len(result) == 2
    assert result[0] == "Response to: First prompt: Hello"
    assert (
        result[1]
        == "Response to: Second prompt: World and Response to: First prompt: Hello"
    )


def test_chainable_json_output():
    # Mock model and callable function
    class MockModel:
        pass

    def mock_callable_prompt(model, prompt):
        if "Output JSON" in prompt:
            return '{"key": "value"}'
        return prompt

    # Test context and chains
    context = {"test": "JSON"}
    chains = ["Output JSON: {{test}}", "Reference JSON: {{output[-1].key}}"]

    # Run the Chainable
    result, _ = MinimalChainable.run(context, MockModel(), mock_callable_prompt, chains)

    # Assert the results
    assert len(result) == 2
    assert isinstance(result[0], dict)
    print("result", result)
    assert result[0] == {"key": "value"}
    assert result[1] == "Reference JSON: value"  # Remove quotes around "value"


def test_chainable_reference_entire_json_output():
    # Mock model and callable function
    class MockModel:
        pass

    def mock_callable_prompt(model, prompt):
        if "Output JSON" in prompt:
            return '{"key": "value"}'
        return prompt

    context = {"test": "JSON"}
    chains = ["Output JSON: {{test}}", "Reference JSON: {{output[-1]}}"]

    # Run the Chainable
    result, _ = MinimalChainable.run(context, MockModel(), mock_callable_prompt, chains)

    assert len(result) == 2
    assert isinstance(result[0], dict)
    assert result[0] == {"key": "value"}
    assert result[1] == 'Reference JSON: {"key": "value"}'


def test_chainable_reference_long_output_value():
    # Mock model and callable function
    class MockModel:
        pass

    def mock_callable_prompt(model, prompt):
        return prompt

    context = {"test": "JSON"}
    chains = [
        "Output JSON: {{test}}",
        "1 Reference JSON: {{output[-1]}}",
        "2 Reference JSON: {{output[-2]}}",
        "3 Reference JSON: {{output[-1]}}",
    ]

    # Run the Chainable
    result, _ = MinimalChainable.run(context, MockModel(), mock_callable_prompt, chains)

    assert len(result) == 4
    assert result[0] == "Output JSON: JSON"
    assert result[1] == "1 Reference JSON: Output JSON: JSON"
    assert result[2] == "2 Reference JSON: Output JSON: JSON"
    assert result[3] == "3 Reference JSON: 2 Reference JSON: Output JSON: JSON"


def test_chainable_empty_context():
    # Mock model and callable function
    class MockModel:
        pass

    def mock_callable_prompt(model, prompt):
        return prompt

    # Test with empty context
    context = {}
    chains = ["Simple prompt"]

    # Run the Chainable
    result, _ = MinimalChainable.run(context, MockModel(), mock_callable_prompt, chains)

    # Assert the results
    assert len(result) == 1
    assert result[0] == "Simple prompt"


def test_chainable_json_output_with_markdown():
    # Mock model and callable function
    class MockModel:
        pass

    def mock_callable_prompt(model, prompt):
        return """
        Here's a JSON response wrapped in markdown:
        ```json
        {
            "key": "value",
            "number": 42,
            "nested": {
                "inner": "content"
            }
        }
        ```
        """

    context = {}
    chains = ["Test JSON parsing"]

    # Run the Chainable
    result, _ = MinimalChainable.run(context, MockModel(), mock_callable_prompt, chains)

    # Assert the results
    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert result[0] == {"key": "value", "number": 42, "nested": {"inner": "content"}}


def test_fusion_chain_run():
    # Mock models
    class MockModel:
        def __init__(self, name):
            self.name = name

    # Mock callable function
    def mock_callable_prompt(model, prompt):
        return f"{model.name} response: {prompt}"

    # Mock evaluator function (random scores between 0 and 1)
    def mock_evaluator(outputs):
        top_response = random.choice(outputs)
        scores = [random.random() for _ in outputs]
        return top_response, scores

    # Test context and chains
    context = {"var1": "Hello", "var2": "World"}
    chains = ["First prompt: {{var1}}", "Second prompt: {{var2}} and {{output[-1]}}"]

    # Create mock models
    models = [MockModel(f"Model{i}") for i in range(3)]

    # Mock get_model_name function
    def mock_get_model_name(model):
        return model.name

    # Run the FusionChain
    result = FusionChain.run(
        context=context,
        models=models,
        callable=mock_callable_prompt,
        prompts=chains,
        evaluator=mock_evaluator,
        get_model_name=mock_get_model_name,
    )

    # Assert the results
    assert isinstance(result, FusionChainResult)
    assert len(result.all_prompt_responses) == 3
    assert len(result.all_context_filled_prompts) == 3
    assert len(result.performance_scores) == 3
    assert len(result.model_names) == 3

    for i, (outputs, context_filled_prompts) in enumerate(
        zip(result.all_prompt_responses, result.all_context_filled_prompts)
    ):
        assert len(outputs) == 2
        assert len(context_filled_prompts) == 2

        assert outputs[0] == f"Model{i} response: First prompt: Hello"
        assert (
            outputs[1]
            == f"Model{i} response: Second prompt: World and Model{i} response: First prompt: Hello"
        )

        assert context_filled_prompts[0] == "First prompt: Hello"
        assert (
            context_filled_prompts[1]
            == f"Second prompt: World and Model{i} response: First prompt: Hello"
        )

    # Check that performance scores are between 0 and 1
    assert all(0 <= score <= 1 for score in result.performance_scores)

    # Check that the number of unique scores is likely more than 1 (random function)
    assert (
        len(set(result.performance_scores)) > 1
    ), "All performance scores are the same, which is unlikely with a random evaluator"

    # Check that top_response is present and is either a string or a dict
    assert isinstance(result.top_response, (str, dict))

    # Print the output of FusionChain.run
    print("All outputs:")
    for i, outputs in enumerate(result.all_prompt_responses):
        print(f"Model {i}:")
        for j, output in enumerate(outputs):
            print(f"  Chain {j}: {output}")

    print("\nAll context filled prompts:")
    for i, prompts in enumerate(result.all_context_filled_prompts):
        print(f"Model {i}:")
        for j, prompt in enumerate(prompts):
            print(f"  Chain {j}: {prompt}")

    print("\nPerformance scores:")
    for i, score in enumerate(result.performance_scores):
        print(f"Model {i}: {score}")

    print("\nTop response:")
    print(result.top_response)

    print("result.model_dump: ", result.model_dump())
    print("result.model_dump_json: ", result.model_dump_json())
