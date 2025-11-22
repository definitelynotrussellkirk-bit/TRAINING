#!/usr/bin/env python3
"""
Interactive Training Mode - Full User Control

This interactive mode:
1. Asks you which generator to use
2. Asks for any settings you want to change
3. Asks how much data to generate
4. Pre-generates ALL the data to a permanent file
5. Validates the generated data
6. Trains the model
7. Asks if you want to delete the training data (doesn't do it automatically)

Perfect for:
- Experimenting with different generators
- Keeping generated data for inspection
- Full control over the workflow

Usage:
    python3 interactive_train.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
LEO_PATH = Path("/media/user/ST/aiPROJECT/LEO")
sys.path.append(str(LEO_PATH))

from validator import DatasetValidator
from train import UltimateTrainer


# Available generators
GENERATORS = {
    "test": {
        "name": "Test Generator (Simplest!)",
        "description": "Ultra-simple counting generator - perfect for quick testing",
        "settings": {
            "type": {
                "type": "choice",
                "prompt": "Generator type",
                "choices": ["counting", "math"],
                "default": "counting"
            }
        }
    },
    "mini": {
        "name": "Property Count (Mini)",
        "description": "Simple property counting tasks - good for testing",
        "settings": {}
    },
    "leo_property": {
        "name": "LEO Property Count",
        "description": "Full LEO property counting with real catalog",
        "settings": {}
    }
}


def print_header(title):
    """Print a nice header."""
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print()


def print_section(title):
    """Print a section divider."""
    print()
    print("─" * 80)
    print(f"  {title}")
    print("─" * 80)
    print()


def ask_yes_no(prompt, default="yes"):
    """Ask a yes/no question."""
    choices = "Y/n" if default == "yes" else "y/N"
    while True:
        response = input(f"{prompt} [{choices}]: ").strip().lower()
        if not response:
            return default == "yes"
        if response in ["y", "yes"]:
            return True
        if response in ["n", "no"]:
            return False
        print("Please enter 'yes' or 'no'")


def ask_number(prompt, default, min_val=None, max_val=None):
    """Ask for a number with validation."""
    while True:
        response = input(f"{prompt} [default: {default}]: ").strip()
        if not response:
            return default
        try:
            value = int(response)
            if min_val is not None and value < min_val:
                print(f"Value must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be at most {max_val}")
                continue
            return value
        except ValueError:
            print("Please enter a valid number")


def ask_float(prompt, default, min_val=None, max_val=None):
    """Ask for a float with validation."""
    while True:
        response = input(f"{prompt} [default: {default}]: ").strip()
        if not response:
            return default
        try:
            value = float(response)
            if min_val is not None and value < min_val:
                print(f"Value must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be at most {max_val}")
                continue
            return value
        except ValueError:
            print("Please enter a valid number")


def ask_string(prompt, default):
    """Ask for a string."""
    response = input(f"{prompt} [default: {default}]: ").strip()
    return response if response else default


def select_generator():
    """Let user select a generator."""
    print_section("STEP 1: Select Generator")

    print("Available generators:")
    print()
    for i, (key, info) in enumerate(GENERATORS.items(), 1):
        print(f"{i}. {info['name']}")
        print(f"   {info['description']}")
        print()

    while True:
        try:
            choice = int(input(f"Select generator [1-{len(GENERATORS)}]: ").strip())
            if 1 <= choice <= len(GENERATORS):
                selected_key = list(GENERATORS.keys())[choice - 1]
                return selected_key, GENERATORS[selected_key]
            else:
                print(f"Please enter a number between 1 and {len(GENERATORS)}")
        except ValueError:
            print("Please enter a valid number")


def configure_generator_settings(generator_key, generator_info):
    """Ask user for generator-specific settings."""
    print_section("STEP 2: Generator Settings")

    if not generator_info["settings"]:
        print("This generator has no configurable settings.")
        return {}

    settings = {}
    for key, config in generator_info["settings"].items():
        if config["type"] == "int":
            settings[key] = ask_number(
                config["prompt"],
                config["default"],
                config.get("min"),
                config.get("max")
            )
        elif config["type"] == "float":
            settings[key] = ask_float(
                config["prompt"],
                config["default"],
                config.get("min"),
                config.get("max")
            )
        elif config["type"] == "choice":
            # Handle choice settings
            print(f"{config['prompt']}:")
            for i, choice in enumerate(config["choices"], 1):
                print(f"  {i}. {choice}")
            while True:
                try:
                    choice_idx = int(input(f"Select [1-{len(config['choices'])}] (default: {config['default']}): ").strip() or "1")
                    if 1 <= choice_idx <= len(config['choices']):
                        settings[key] = config["choices"][choice_idx - 1]
                        break
                except ValueError:
                    pass
                print(f"Please enter 1-{len(config['choices'])}")
        else:
            settings[key] = ask_string(config["prompt"], config["default"])

    return settings


def configure_training_settings():
    """Ask user for training settings."""
    print_section("STEP 3: Training Configuration")

    print("Let's configure the training parameters.")
    print("Press Enter to use defaults (shown in brackets).")
    print()

    config = {}

    # Number of samples
    config["num_samples"] = ask_number(
        "How many training samples to generate?",
        default=1000,
        min_val=10
    )

    # Model
    config["model"] = ask_string(
        "Model name or path",
        default="qwen3_0.6b"
    )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"~/interactive_adapter_{timestamp}"
    config["output_dir"] = ask_string(
        "Output directory for trained adapter",
        default=default_output
    )

    # Ask if user wants to customize training params
    print()
    if ask_yes_no("Customize advanced training parameters?", default="no"):
        print()
        config["epochs"] = ask_number("Number of epochs", default=2, min_val=1)
        config["batch_size"] = ask_number("Batch size", default=4, min_val=1)
        config["gradient_accumulation"] = ask_number(
            "Gradient accumulation steps",
            default=4,
            min_val=1
        )
        config["learning_rate"] = ask_float(
            "Learning rate",
            default=2e-4,
            min_val=1e-6,
            max_val=1e-3
        )
    else:
        # Use defaults
        config["epochs"] = 2
        config["batch_size"] = 4
        config["gradient_accumulation"] = 4
        config["learning_rate"] = 2e-4

    return config


def create_generator_function(generator_key, settings):
    """Create the actual generator function."""

    if generator_key == "test":
        # Use simplest test generator
        def test_generator():
            sys.path.append(str(Path(__file__).parent / "examples"))
            from simplest_generator import simple_counting_generator, simple_math_generator

            gen_type = settings.get("type", "counting")
            if gen_type == "counting":
                gen = simple_counting_generator()
            else:
                gen = simple_math_generator()

            for record in gen:
                yield record

        return test_generator

    elif generator_key == "mini":
        # Use mini generator
        def mini_generator():
            sys.path.append(str(Path(__file__).parent / "examples"))
            from leo_mini_generator import generate_property_count_sample

            i = 0
            while True:
                record = generate_property_count_sample(i)
                yield {"messages": record["messages"]}
                i += 1

        return mini_generator

    elif generator_key == "leo_property":
        # Use real LEO generator
        def leo_generator():
            try:
                from LEO.HER.generators import property_count
                from LEO.PAR.catalog_cache import load_items

                items = load_items()
                i = 0

                while True:
                    import random
                    rng = random.Random(i)

                    # Sample items
                    selected = rng.sample(items, min(7, len(items)))

                    # Choose property
                    from LEO.HER.generators.property_count import choose_property
                    prop = choose_property(rng, selected)

                    # Build record
                    from LEO.HER.generators.property_count import build_record
                    from LEO.HER.generators.answer_forms import count_answer_form
                    from LEO.HER.generators.instruction_builder import build_instruction

                    answer_form = count_answer_form()
                    instruction = build_instruction(
                        skill_id="list.count_property_matches",
                        variables={"property_id": prop},
                        template_index=i % 5
                    )

                    record = build_record(
                        items=items,
                        selected_items=selected,
                        property_id=prop,
                        answer_instructions=answer_form["instructions"],
                        answer_form=answer_form,
                        instruction_text=instruction,
                        instruction_template=f"template_{i % 5}"
                    )

                    yield {
                        "messages": [
                            {"role": "user", "content": record["prompt"]},
                            {"role": "assistant", "content": record["response"]}
                        ]
                    }

                    i += 1

            except ImportError as e:
                print(f"❌ Error: LEO not found at {LEO_PATH}")
                print(f"   {e}")
                print()
                print("Falling back to mini generator...")

                sys.path.append(str(Path(__file__).parent / "examples"))
                from leo_mini_generator import generate_property_count_sample

                i = 0
                while True:
                    record = generate_property_count_sample(i)
                    yield {"messages": record["messages"]}
                    i += 1

        return leo_generator

    else:
        raise ValueError(f"Unknown generator: {generator_key}")


def generate_data_to_file(generator_func, num_samples, output_file):
    """Generate data and save to permanent file."""
    print_section("STEP 4: Generating Data")

    print(f"Generating {num_samples:,} samples to: {output_file}")
    print()

    gen = generator_func()
    count = 0

    with open(output_file, 'w') as f:
        for i, record in enumerate(gen):
            if i >= num_samples:
                break

            f.write(json.dumps(record) + '\n')
            count += 1

            # Progress indicator
            if (i + 1) % 100 == 0:
                pct = (i + 1) / num_samples * 100
                print(f"  Generated {i + 1:,} / {num_samples:,} samples ({pct:.1f}%)")

    print()
    print(f"✅ Generated {count:,} samples")

    # Show file stats
    file_size = Path(output_file).stat().st_size / 1024 / 1024
    print(f"   File: {output_file}")
    print(f"   Size: {file_size:.2f} MB")
    print()

    return output_file


def validate_data_file(data_file):
    """Validate the generated data."""
    print_section("STEP 5: Validating Data")

    print("Running validation checks...")
    print()

    validator = DatasetValidator(Path(data_file))
    passed = validator.run_full_validation()

    if not passed:
        print()
        print("❌ Validation failed!")
        print()
        print("The data has issues. Please review the errors above.")
        print()

        if not ask_yes_no("Continue with training anyway? (NOT recommended)", default="no"):
            return False
    else:
        print()
        print("✅ Validation passed! Data looks good.")

    return True


def train_model(data_file, config):
    """Train the model."""
    print_section("STEP 6: Training Model")

    print("Starting training with Ultimate Trainer...")
    print()

    # Confirm before training
    print("Training configuration:")
    print(f"  Dataset: {data_file}")
    print(f"  Model: {config['model']}")
    print(f"  Output: {config['output_dir']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print()

    if not ask_yes_no("Start training now?", default="yes"):
        print("Training cancelled.")
        return False

    # Create args for UltimateTrainer
    class Args:
        pass

    args = Args()
    args.dataset = str(data_file)
    args.model = config["model"]
    args.output_dir = config["output_dir"]
    args.epochs = config["epochs"]
    args.batch_size = config["batch_size"]
    args.gradient_accumulation = config["gradient_accumulation"]
    args.learning_rate = config["learning_rate"]
    args.warmup_steps = 100
    args.lora_r = 64
    args.lora_alpha = 32
    args.eval_steps = 100
    args.num_eval_samples = 5
    args.save_steps = 500
    args.skip_validation = True  # Already validated
    args.yes = True  # Skip confirmations in trainer

    # Run trainer
    trainer = UltimateTrainer(args)
    success = trainer.run()

    return success


def cleanup_prompt(data_file):
    """Ask user if they want to delete the training data."""
    print_section("STEP 7: Cleanup")

    file_size = Path(data_file).stat().st_size / 1024 / 1024

    print(f"Training data file: {data_file}")
    print(f"Size: {file_size:.2f} MB")
    print()
    print("Options:")
    print("  - Keep it: Useful for inspection, retraining, or sharing")
    print("  - Delete it: Free up disk space")
    print()

    if ask_yes_no("Delete the training data file?", default="no"):
        try:
            Path(data_file).unlink()
            print(f"✅ Deleted: {data_file}")
        except Exception as e:
            print(f"❌ Error deleting file: {e}")
    else:
        print(f"✅ Kept training data: {data_file}")


def main():
    """Interactive training workflow."""

    print_header("🤖 Interactive Training Mode - Full Control")

    print("This interactive workflow will:")
    print("  1. Ask which generator you want to use")
    print("  2. Ask for any settings you want to change")
    print("  3. Ask how much data to generate")
    print("  4. Pre-generate ALL the data to a file")
    print("  5. Validate the data")
    print("  6. Train the model")
    print("  7. Ask if you want to delete the training data")
    print()

    if not ask_yes_no("Ready to begin?", default="yes"):
        print("Cancelled.")
        return

    # Step 1: Select generator
    generator_key, generator_info = select_generator()
    print(f"✅ Selected: {generator_info['name']}")

    # Step 2: Configure generator settings (if any)
    gen_settings = configure_generator_settings(generator_key, generator_info)

    # Step 3: Configure training
    config = configure_training_settings()
    print()
    print("✅ Configuration complete")

    # Create output filename for generated data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = Path(f"/tmp/interactive_training_data_{timestamp}.jsonl")

    # Step 4: Generate data
    generator_func = create_generator_function(generator_key, gen_settings)
    generate_data_to_file(generator_func, config["num_samples"], data_file)

    # Step 5: Validate
    if not validate_data_file(data_file):
        print("Aborting due to validation failure.")
        return

    # Step 6: Train
    success = train_model(data_file, config)

    if not success:
        print()
        print("❌ Training failed!")
        print()

        # Still ask about cleanup
        cleanup_prompt(data_file)
        return

    # Step 7: Cleanup
    print()
    print("=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print(f"✅ Model saved to: {config['output_dir']}")
    print()

    cleanup_prompt(data_file)

    print()
    print("=" * 80)
    print("All done! 🚀")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print()
        print("Interrupted by user.")
        sys.exit(1)
