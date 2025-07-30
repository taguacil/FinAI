#!/usr/bin/env python3
"""
Main entry point for the Portfolio Tracker application.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.initializer import PortfolioInitializer


def main():
    """Main function to initialize and run the portfolio tracker."""
    parser = argparse.ArgumentParser(description="AI Portfolio Tracker")
    parser.add_argument("--mode", choices=["ui", "init", "sample", "status"], 
                       default="ui", help="Application mode")
    parser.add_argument("--data-dir", default="data", 
                       help="Data directory path")
    parser.add_argument("--sample-name", default="Demo Portfolio",
                       help="Name for sample portfolio")
    
    args = parser.parse_args()
    
    # Initialize the system
    initializer = PortfolioInitializer(args.data_dir)
    
    if args.mode == "init":
        print("🚀 Initializing Portfolio Tracker system...")
        results = initializer.initialize_system()
        
        print("\n📊 Initialization Results:")
        for component, status in results.items():
            status_emoji = "✅" if status else "❌"
            print(f"  {status_emoji} {component.replace('_', ' ').title()}: {status}")
        
        if all(results.values()):
            print("\n🎉 System initialization completed successfully!")
        else:
            print("\n⚠️  Some components failed to initialize. Check logs for details.")
    
    elif args.mode == "sample":
        print(f"📝 Creating sample portfolio: {args.sample_name}")
        portfolio_id = initializer.create_sample_portfolio(args.sample_name)
        
        if portfolio_id:
            print(f"✅ Sample portfolio created successfully!")
            print(f"   Portfolio ID: {portfolio_id}")
            print(f"   You can now run the UI to interact with your portfolio.")
        else:
            print("❌ Failed to create sample portfolio. Check logs for details.")
    
    elif args.mode == "status":
        print("📊 Getting system status...")
        status = initializer.get_system_status()
        
        if 'error' in status:
            print(f"❌ Error getting status: {status['error']}")
            return
        
        print(f"\n🕒 Status as of: {status['timestamp']}")
        print(f"\n📁 Storage:")
        print(f"   Data Directory: {status['storage']['data_directory']}")
        print(f"   Directory Exists: {status['storage']['directories_exist']}")
        
        print(f"\n🔌 Data Providers:")
        for provider, working in status['data_providers'].items():
            status_emoji = "✅" if working else "❌"
            print(f"   {status_emoji} {provider}")
        
        print(f"\n💼 Portfolios: {status['portfolios']['total_count']} total")
        for portfolio in status['portfolios']['portfolio_list']:
            print(f"   📊 {portfolio['name']} ({portfolio['id'][:8]}...)")
            print(f"      Created: {portfolio['created']}")
            print(f"      Positions: {portfolio['positions']}")
            print(f"      Snapshots: {portfolio['snapshots']}")
            print(f"      Last Update: {portfolio['last_snapshot'] or 'Never'}")
            print()
    
    elif args.mode == "ui":
        print("🚀 Starting Portfolio Tracker UI...")
        
        # Initialize system first
        print("   Initializing system...")
        init_results = initializer.initialize_system()
        
        # Check if we have any portfolios
        portfolios = initializer.storage.list_portfolios()
        if not portfolios:
            print("\n📝 No portfolios found. Creating a sample portfolio...")
            sample_id = initializer.create_sample_portfolio("My Portfolio")
            if sample_id:
                print(f"✅ Sample portfolio created: {sample_id}")
            else:
                print("❌ Failed to create sample portfolio")
        
        # Start the UI
        try:
            from src.ui.streamlit_app import main as run_streamlit
            print("🌐 Starting web interface...")
            print("   Open your browser to: http://localhost:8501")
            run_streamlit()
        except ImportError as e:
            print(f"❌ Failed to start UI: {e}")
            print("   Make sure all dependencies are installed: uv sync")
        except Exception as e:
            print(f"❌ Error starting UI: {e}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()