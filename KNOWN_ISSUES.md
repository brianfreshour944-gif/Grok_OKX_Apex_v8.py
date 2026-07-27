# Known Issues / Follow-Up Work

Last updated: 2026-07-27

## Run signal_ic_check.py against real historical data

This diagnostic tool exists and works (fixed from a broken filename in
commit c7283ac), but has apparently never been run against this
specific model. It tests whether the model's raw prediction actually
correlates with future price movement (walk-forward Information
Coefficient) - the check that should happen before trusting any
threshold/exit tuning to matter. Given a sibling model in this same
lineage (Apex_oracle_bot repo) was found to have a real feature-scaling
bug that was crushing most of its predictive signal, it's worth running
this check here too to know whether this model has genuine edge.

Usage: python3 signal_ic_check.py --csv-dir historical_data/

## Symbol format inconsistency (unverified)

orders.py's place_order() gets called with slash-format symbols
("BTC/USD") from the normal entry/exit flow in main_bot.py, but with
no-slash format ("BTCUSD") from portfolio.py's sell_largest_position()
(which reads the format from Alpaca position objects). Both currently
go through the same LimitOrderRequest call. This may be fine - Alpaca's
API has historically been inconsistent about which format different
endpoints expect - but hasn't been directly verified. Worth a live test:
trigger the portfolio-cap forced-sell path and confirm the order
succeeds without a symbol-format error.

## Exit logic has no test coverage

main_bot.py's exit logic (trailing stop, time-decay stop loss, max
hold time, signal-weak exit) is sophisticated but untested. A handful
of unit tests feeding known price/signal sequences and asserting the
correct exit_reason fires would catch regressions early - similar in
spirit to what test_overfit.py already does for the model architecture
itself (in scripts/).
