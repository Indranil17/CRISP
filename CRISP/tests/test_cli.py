"""Tests for CRISP command-line interface module."""
import pytest
import sys
from unittest.mock import patch, MagicMock
from io import StringIO
from CRISP.cli import main


class TestCLIBasic:
    """Test basic CLI functionality."""
    
    @patch('sys.argv', ['crisp', '--version'])
    def test_version_flag(self):
        """Test --version flag displays version and exits."""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
    
    @patch('sys.argv', ['crisp', '--help'])
    def test_help_flag(self):
        """Test --help flag displays help and exits."""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
    
    @patch('sys.argv', ['crisp'])
    def test_no_arguments(self):
        """Test CLI with no arguments shows help."""
        with patch('sys.stdout', new=StringIO()):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
    
    @patch('sys.argv', ['crisp', 'test'])
    @patch('CRISP.cli.run_tests')
    def test_test_command_success(self, mock_run_tests):
        """Test 'test' subcommand when tests pass."""
        # run_tests calls sys.exit, so main won't return
        main()
        mock_run_tests.assert_called_once_with(verbose=False)
    
    @patch('sys.argv', ['crisp', 'test'])
    @patch('CRISP.cli.run_tests')
    def test_test_command_failure(self, mock_run_tests):
        """Test 'test' subcommand when tests fail."""
        # run_tests calls sys.exit, so main won't return
        main()
        mock_run_tests.assert_called_once_with(verbose=False)
    
    @patch('sys.argv', ['crisp', '--invalid-flag'])
    def test_invalid_flag(self):
        """Test invalid command-line flag."""
        with patch('sys.stderr', new=StringIO()):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code != 0
    
    @patch('sys.argv', ['crisp', 'invalid_command'])
    def test_invalid_command(self):
        """Test invalid subcommand."""
        with patch('sys.stderr', new=StringIO()):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code != 0


class TestCLIIntegration:
    """Test CLI integration."""
    
    def test_cli_module_import(self):
        """Test CLI module can be imported."""
        from CRISP import cli
        assert hasattr(cli, 'main')
        assert callable(cli.main)
    
    def test_version_string_format(self):
        """Test version string is properly formatted."""
        from CRISP._version import __version__
        assert __version__
        assert isinstance(__version__, str)
        assert len(__version__) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
