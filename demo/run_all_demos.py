"""
Master Demo Runner
Executes all pipeline component demos in sequence to provide a comprehensive demonstration.
"""

import sys
import os
import importlib.util

def run_demo_module(demo_path, demo_name):
    """Import a demo module by path and execute it in-process.

    Parameters
    ----------
    demo_path : str
        Filesystem path to the demo script to run.
    demo_name : str
        Friendly name used for framing console output.

    Returns
    -------
    bool
        ``True`` when execution succeeds, ``False`` on exception.
    """
    print(f"\n{'='*80}")
    print(f"EXECUTING: {demo_name}")
    print(f"{'='*80}")

    try:
        spec = importlib.util.spec_from_file_location(demo_name, demo_path)
        demo_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo_module)
        print(f"✅ {demo_name} completed successfully!")
        return True
    except Exception as e:
        print(f"❌ {demo_name} failed: {e}")
        return False

def main():
    """Iterate through all demo modules, prompting between each run."""
    print("🚀 QUERY PIPELINE COMPREHENSIVE DEMONSTRATION SUITE")
    print("=" * 80)
    print("This master demo runs all pipeline components to demonstrate:")
    print("• Embedder initialization and IRI mapping")
    print("• Query-to-embedding matching")
    print("• Clause extraction and compression")
    print("• Prompt template processing")
    print("• LLM interaction simulation")
    print("• GraphDB query execution")
    print("• Error handling and retry logic")
    print("• Complete end-to-end pipeline integration")
    print("=" * 80)

    # Demo modules in execution order
    demo_modules = [
        ("demo_01_embedder_initialization.py", "Embedder Initialization"),
        ("demo_02_embedding_matching.py", "Embedding Matching & IRI Extraction"),
        ("demo_03_clause_extraction.py", "Clause Extraction & Chunking"),
        ("demo_04_prompt_building.py", "Prompt Building & Template Processing"),
        ("demo_05_llm_interaction.py", "LLM Interaction & SPARQL Generation"),
        ("demo_06_graphdb_query.py", "GraphDB Query Execution"),
        ("demo_07_error_handling.py", "Error Handling & Retry Mechanisms"),
        ("demo_08_complete_pipeline.py", "Complete Pipeline Integration")
    ]

    successful_demos = 0
    total_demos = len(demo_modules)

    user_input = input("\n🤖 Press Enter to start the demonstrations, or 'q' to quit: ")
    if user_input.lower() == 'q':
        print("Demo cancelled by user.")
        return

    for demo_file, demo_name in demo_modules:
        demo_path = os.path.join(os.path.dirname(__file__), demo_file)

        if os.path.exists(demo_path):
            success = run_demo_module(demo_path, demo_name)
            if success:
                successful_demos += 1
        else:
            print(f"❌ Demo file not found: {demo_file}")

        # Pause between demos for readability
        input(f"\n⏸️  Press Enter to continue to the next demo...")

    # Final summary
    print(f"\n{'='*80}")
    print(f"DEMONSTRATION SUITE COMPLETED")
    print(f"{'='*80}")
    print(f"📊 Success Rate: {successful_demos}/{total_demos} demos completed successfully")
    print(f"📁 Output files have been generated in the demo/ directory")
    print(f"🔍 Check individual demo outputs for detailed component analysis")

    if successful_demos == total_demos:
        print(f"🎉 All demonstrations completed successfully!")
        print(f"   The query pipeline and all its components are working correctly.")
    else:
        print(f"⚠️  Some demonstrations failed. Check error messages above.")
        print(f"   This may indicate missing dependencies or configuration issues.")

    print(f"\n📚 For detailed documentation, see README_DEMOS.md")
    print(f"💡 To run individual demos: python demo_XX_component_name.py")

if __name__ == "__main__":
    main()
