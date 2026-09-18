# Utils module for shared methods
import os, subprocess

def assertHandle(func, *args, cwd='', message=''):
  """
  ### This function runs the given assertion and handles the potential error, showing the protocol's error trace instead of a generic assertion.
  ### Note: Only designed to handle the assertions used by protocols.

  #### Params:
  - func (function): A function (assertion) to be executed.
  - *args: All the arguments passed to the function.
  - cwd (str): Optional. Current working directory. Usually protocol's cwd.
  - message (str): Optional. Custom message to display if no error messages were found in stdout/stderr.

  #### Example:
  assertHandle(self.assertIsNotNone, getattr(protocol, 'outputSet', None), cwd=protocol.getWorkingDir())
  """
  # Defining full path to error log
  stderr = os.path.abspath(os.path.join(cwd, 'logs', 'run.stderr'))
  stdout = os.path.abspath(os.path.join(cwd, 'logs', 'run.stdout'))

  # Attempt to run assertion
  try:
    return func(*args)
  except AssertionError:
    # If assertion fails, show error log
    # Getting error logs (stderr has priority over stdout)
    # Most errors are dumped on stderr, while some others on stdout
    errorMessage = ''