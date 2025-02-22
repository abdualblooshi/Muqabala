# Fix torch path issue with Streamlit
import os
import torch
torch.classes.__path__ = []  # Fix for torch path warning in Streamlit

import logging
import threading
import time
import queue
import streamlit as st
from functools import wraps
from typing import Optional, Callable, Any
import traceback
import sys
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('interview_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Filter PyTorch warnings
class TorchWarningFilter(logging.Filter):
    def filter(self, record):
        return not (
            "Tried to instantiate class '__path__._path'" in str(record.msg) or
            "torch.classes.__path__" in str(record.msg)
        )

# Add filter to logger
logger.addFilter(TorchWarningFilter())

# Suppress specific PyTorch warnings
warnings.filterwarnings("ignore", message=".*Tried to instantiate class '__path__._path'.*")
warnings.filterwarnings("ignore", message=".*torch.classes.__path__.*")

class ThreadMonitor:
    """Monitor and manage threads with detailed logging."""
    
    def __init__(self):
        self.active_threads = {}
        self.thread_logs = queue.Queue()
        self._monitor_thread = None
        self._stop_monitor = False
    
    def start_monitoring(self):
        """Start the thread monitoring service."""
        if not self._monitor_thread:
            self._stop_monitor = False
            self._monitor_thread = threading.Thread(target=self._monitor_threads)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            logger.info("Thread monitoring started")
    
    def stop_monitoring(self):
        """Stop the thread monitoring service."""
        if self._monitor_thread:
            self._stop_monitor = True
            self._monitor_thread.join()
            self._monitor_thread = None
            logger.info("Thread monitoring stopped")
    
    def _monitor_threads(self):
        """Monitor thread activity and log status changes."""
        while not self._stop_monitor:
            current_threads = threading.enumerate()
            for thread in current_threads:
                thread_id = thread.ident
                if thread_id not in self.active_threads:
                    self.active_threads[thread_id] = {
                        'name': thread.name,
                        'start_time': time.time()
                    }
                    logger.info(f"New thread started: {thread.name} (ID: {thread_id})")
            
            # Check for finished threads
            finished_threads = []
            for thread_id in self.active_threads:
                if not any(t.ident == thread_id for t in current_threads):
                    thread_info = self.active_threads[thread_id]
                    duration = time.time() - thread_info['start_time']
                    logger.info(f"Thread finished: {thread_info['name']} (ID: {thread_id}, Duration: {duration:.2f}s)")
                    finished_threads.append(thread_id)
            
            # Remove finished threads from tracking
            for thread_id in finished_threads:
                del self.active_threads[thread_id]
            
            time.sleep(0.1)  # Prevent excessive CPU usage

class StreamlitContextManager:
    """Manage Streamlit context in threaded environments."""
    
    @staticmethod
    def ensure_context(func: Callable) -> Callable:
        """Decorator to ensure Streamlit context is available."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            import streamlit.runtime.scriptrunner.script_runner as script_runner
            ctx = script_runner.get_script_run_ctx()
            if ctx:
                script_runner.add_script_run_ctx(ctx)
            return func(*args, **kwargs)
        return wrapper
    
    @staticmethod
    def safe_session_state_update(key: str, value: Any) -> bool:
        """Safely update Streamlit session state."""
        try:
            st.session_state[key] = value
            return True
        except Exception as e:
            logger.error(f"Failed to update session state key '{key}': {str(e)}")
            return False

class DebugManager:
    """Manage debug state and provide debugging utilities."""
    
    def __init__(self):
        self.thread_monitor = ThreadMonitor()
        self.debug_queue = queue.Queue()
        self.is_debug_mode = False
        self._setup_torch_error_handling()
    
    def _setup_torch_error_handling(self):
        """Configure PyTorch-specific error handling."""
        try:
            def torch_error_handler(error_type, error_msg, *args):
                if "Tried to instantiate class '__path__._path'" in str(error_msg):
                    if self.is_debug_mode:
                        logger.debug(f"Suppressed PyTorch warning: {error_msg}")
                    return
                logger.error(f"PyTorch error: {error_type} - {error_msg}")
            
            # Only set error handler if the attribute exists (PyTorch >= 2.0)
            if hasattr(torch, '_C') and hasattr(torch._C, '_set_error_handler'):
                torch._C._set_error_handler(torch_error_handler)
        except Exception as e:
            logger.warning(f"Could not set PyTorch error handler: {e}")
            # Continue without custom error handling
    
    def start_debug_mode(self):
        """Enable debug mode with enhanced logging and monitoring."""
        self.is_debug_mode = True
        self.thread_monitor.start_monitoring()
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
        self._setup_torch_error_handling()  # Reconfigure error handling
    
    def stop_debug_mode(self):
        """Disable debug mode."""
        self.is_debug_mode = False
        self.thread_monitor.stop_monitoring()
        logger.setLevel(logging.INFO)
        logger.info("Debug mode disabled")
    
    def log_error(self, error: Exception, context: Optional[str] = None):
        """Log an error with full stack trace and context."""
        # Skip logging for suppressed PyTorch warnings
        if isinstance(error, Warning) and "Tried to instantiate class '__path__._path'" in str(error):
            if self.is_debug_mode:
                logger.debug(f"Suppressed PyTorch warning in {context}")
            return
        
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'stack_trace': traceback.format_exc(),
            'context': context,
            'timestamp': time.time()
        }
        logger.error(f"Error occurred: {error_info['error_type']}")
        logger.error(f"Message: {error_info['error_message']}")
        logger.error(f"Context: {error_info['context']}")
        logger.error(f"Stack trace:\n{error_info['stack_trace']}")
        self.debug_queue.put(error_info)
    
    def get_thread_status(self) -> dict:
        """Get current thread status information."""
        return {
            'active_threads': len(self.thread_monitor.active_threads),
            'thread_details': self.thread_monitor.active_threads.copy()
        }
    
    def get_system_info(self) -> dict:
        """Get system and environment information."""
        return {
            'python_version': sys.version,
            'platform': sys.platform,
            'streamlit_context_available': bool(st.runtime.exists()),
            'environment_variables': dict(os.environ),
            'working_directory': os.getcwd(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available() if hasattr(torch, 'cuda') else False
        }

# Initialize global debug manager
debug_manager = DebugManager()

def debug_wrapper(func: Callable) -> Callable:
    """Decorator to add debug logging to functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.debug(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            debug_manager.log_error(e, f"Error in {func.__name__}")
            raise
    return wrapper

def test_streamlit_context():
    """Test Streamlit context availability and session state access."""
    try:
        # Test session state access
        test_key = "__test_key__"
        test_value = "test_value"
        success = StreamlitContextManager.safe_session_state_update(test_key, test_value)
        if not success:
            logger.warning("Failed to update session state in test")
        
        # Test context availability
        import streamlit.runtime.scriptrunner.script_runner as script_runner
        ctx = script_runner.get_script_run_ctx()
        logger.info(f"Streamlit context available: {bool(ctx)}")
        
        return True
    except Exception as e:
        logger.error(f"Streamlit context test failed: {str(e)}")
        return False 