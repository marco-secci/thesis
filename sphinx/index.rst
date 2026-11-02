Donazioni di sangue Trieste-Gorizia, 2009-2025
==============================================

API documentation of the code behind the thesis: from the 27 raw XLSX files
to the donor x year panel, the models compared in the backtest, the
2026-2031 forecast and the thesis figures.

Every page is generated from the docstrings in the source. The stages run in
this order (``make all`` or ``./run_pipeline.sh``): ``ingest`` → ``clean`` →
``panel`` → analyses and backtests → ``forecast`` → ``figures_tesi`` and
``export_legacy``.

.. toctree::
   :maxdepth: 2
   :caption: Modules

   api/pipeline
   api/analysis
   api/forecasting
   api/output
   api/scripts
   api/tests

Index
-----

* :ref:`genindex`
* :ref:`modindex`
