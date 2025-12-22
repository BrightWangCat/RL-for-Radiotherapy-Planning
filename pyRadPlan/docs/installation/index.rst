.. _installation:

Installation
============

pyRadPlan requires Python 3.9 or higher installed.

From the Python Package Index (PyPI)
------------------------------------

pyRadPlan can easily be installed from pip:

.. code-block:: bash
    pip install pyRadPlan

From source
-----------

Installation from source is mainly useful for developers, and we recommend to work with an editable
installation in a virtual environment.

For developers, we encourage installing the dev dependencies as well:

.. code-block:: bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -e .[dev]

This will install an editable pyRadPlan module including ``pytest`` (with ``coverage`` extensions),
``sphinx`` (+ extensions), and ``pre-commit`` in addition to the standard modules.
- **pytest** is used to run unit tests before publishing the code. Run on the test folder via ``pytest test``, or choose any test file following the pytest syntax. We encourage writing at least fundamental unit tests for new code. Will also install ``coverage`` and the pytest extension to monitor coverage.
- **sphinx** is used to generate the documentation.
- **pre-commit** allows for automatic code formatting to ensure good style and quality within a git pre-commit hook.

After successfully running `pip install -e .[dev]` check in the console that `pre-commit --version` provides a correct response.
Afterwards run `pre-commit install`. This reads the `.pre-commit_config.yaml` and adds a hook to your git repository, which whenever a commit is made the changed files are reformatted to ensure the PEP standards.
