#!/usr/bin/env python3
"""
Main entry point for the Portfolio Tracker application.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.health_check import HealthChecker
from src.utils.initializer import PortfolioInitializer
from src.utils.logging_config import setup_logging


def main():
    """Main function to initialize and run the portfolio tracker."""
    parser = argparse.ArgumentParser(description="AI Portfolio Tracker")
    parser.add_argument(
        "--mode",
        choices=["ui", "init", "sample", "status", "update-snapshots"],
        default="ui",
        help="Application mode",
    )
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument(
        "--sample-name", default="Demo Portfolio", help="Name for sample portfolio"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(log_level=args.log_level, log_dir=f"{args.data_dir}/logs")
    logger.info(f"Starting Portfolio Tracker in {args.mode} mode")

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
        print("📊 Getting comprehensive system status...")

        # Get traditional status
        status = initializer.get_system_status()

        # Get health check results
        health_checker = HealthChecker(args.data_dir)
        health_summary = health_checker.get_health_summary()

        if "error" in status:
            print(f"❌ Error getting status: {status['error']}")
            return

        # Overall health status
        overall_status = health_summary["overall_status"]
        status_emoji = {
            "healthy": "✅",
            "warning": "⚠️",
            "critical": "❌",
            "unknown": "❓",
        }.get(overall_status, "❓")

        print(f"\n🏥 Overall System Health: {status_emoji} {overall_status.upper()}")
        print(f"🕒 Status as of: {status['timestamp']}")

        # Health summary
        summary = health_summary["summary"]
        print(f"\n📊 Health Summary:")
        print(f"   Total Services: {summary['total_services']}")
        print(f"   ✅ Healthy: {summary['healthy_services']}")
        print(f"   ⚠️  Warning: {summary['warning_services']}")
        print(f"   ❌ Critical: {summary['critical_services']}")

        # Service details
        print(f"\n🔧 Service Status:")
        for service_name, service_info in health_summary["services"].items():
            status_emoji = {
                "healthy": "✅",
                "warning": "⚠️",
                "critical": "❌",
                "unknown": "❓",
            }.get(service_info["status"], "❓")

            response_time = ""
            if service_info.get("response_time_ms"):
                response_time = f" ({service_info['response_time_ms']:.0f}ms)"

            print(
                f"   {status_emoji} {service_name.replace('_', ' ').title()}: {service_info['message']}{response_time}"
            )

        print(f"\n📁 Storage:")
        print(f"   Data Directory: {status['storage']['data_directory']}")
        print(f"   Directory Exists: {status['storage']['directories_exist']}")

        print(f"\n🔌 Data Providers:")
        for provider, working in status["data_providers"].items():
            status_emoji = "✅" if working else "❌"
            print(f"   {status_emoji} {provider}")

        print(f"\n💼 Portfolios: {status['portfolios']['total_count']} total")
        for portfolio in status["portfolios"]["portfolio_list"]:
            print(f"   📊 {portfolio['name']} ({portfolio['id'][:8]}...)")
            print(f"      Created: {portfolio['created']}")
            print(f"      Positions: {portfolio['positions']}")
            print(f"      Snapshots: {portfolio['snapshots']}")
            print(f"      Last Update: {portfolio['last_snapshot'] or 'Never'}")
            print()

    elif args.mode == "ui":
        print("🚀 Starting Portfolio Tracker UI...")

        # Initialize system first (load existing data only)
        print("   Loading existing data...")
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
            print("   Note: Use the 'Update Prices' button to fetch new data")
            run_streamlit()
        except ImportError as e:
            print(f"❌ Failed to start UI: {e}")
            print("   Make sure all dependencies are installed: uv sync")
        except Exception as e:
            print(f"❌ Error starting UI: {e}")

    elif args.mode == "update-snapshots":
        print("📊 Updating portfolio snapshots...")

        # Initialize system
        init_results = initializer.initialize_system()
        if not init_results.get("system_ready", False):
            print("❌ System initialization failed")
            return

        # Find and update sample portfolio
        portfolios = initializer.storage.list_portfolios()
        sample_portfolio_id = None

        for portfolio_id in portfolios:
            portfolio = initializer.storage.load_portfolio(portfolio_id)
            if portfolio and (
                "Sample" in portfolio.name or "My Portfolio" in portfolio.name
            ):
                sample_portfolio_id = portfolio_id
                break

        if not sample_portfolio_id:
            print("❌ No sample portfolio found")
            return

        print(f"📈 Updating snapshots for portfolio: {sample_portfolio_id}")

        # Load portfolio and update prices
        portfolio_manager = initializer.portfolio_manager
        portfolio_manager.load_portfolio(sample_portfolio_id)

        # Update current prices
        print("🔄 Updating current market prices...")
        price_results = portfolio_manager.update_current_prices()
        print(f"   Price update results: {price_results}")

        # Create snapshots for the past 60 days
        print("📅 Creating historical snapshots...")
        from datetime import date, timedelta

        end_date = date.today()
        snapshots_created = 0

        for days_back in range(60, -1, -1):
            snapshot_date = end_date - timedelta(days=days_back)
            try:
                snapshot = portfolio_manager.create_snapshot(snapshot_date)
                snapshots_created += 1
                if days_back % 10 == 0:
                    print(f"   Created snapshot for {snapshot_date}")
            except Exception as e:
                print(f"   ⚠️  Failed to create snapshot for {snapshot_date}: {e}")

        print(f"✅ Successfully created {snapshots_created} snapshots")

        # Show current portfolio value
        portfolio_value = portfolio_manager.get_portfolio_value()
        print(f"💰 Current portfolio value: ${portfolio_value:,.2f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
