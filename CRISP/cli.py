"""Command-line interface for CRISP."""

import argparse
import sys
from CRISP._version import __version__


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='crisp',
        description='CRISP: Computational Research Infrastructure for Spectroscopy and Properties'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'CRISP {__version__}'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.command == 'test':
        run_tests(verbose=args.verbose)
    elif args.command is None:
        parser.print_help()
        sys.exit(1)


def run_tests(verbose=False):
    """Run package tests."""
    import pytest
    args = ['-v'] if verbose else []
    sys.exit(pytest.main(args))

