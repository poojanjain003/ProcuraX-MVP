# ProcuraX Refrigerator Demo Dataset — Pricing & Seasonality Notes

## Dataset structure

- 66 refrigerator-component SKUs.
- 2 variants per component family across 33 component families.
- 3 supplier records per SKU (198 supplier rows total).
- Historical demand: 1 Jan 2025 to 10 Sep 2026 (40788 daily SKU observations).
- Procurement category: `Refrigerator Category` for the entire packaged demo set.
- UOM is included for procurement realism (`piece`, `set`, `pair`, or `kg`).

## Demand / forecasting design

The historical series intentionally contains a refrigerator-demand seasonal pattern:
- March: ramp-up
- April: peak
- May: highest peak
- June: taper
- Other months: normalised baseline

The demo is synthetic operational data designed for a hackathon prototype. It is not a claim about any specific OEM's confidential demand plan.

## Online price calibration

Supplier quote anchors were calibrated against publicly listed Indian refrigeration-component prices and market ranges. The app still uses **synthetic supplier quotes** around those benchmarks because public listings are usually retail/small-MOQ or specification-dependent rather than OEM annual-contract RFQs.

Examples used for calibration:

- Refrigerator condenser coils: approximately ₹300/piece for domestic refrigerator coils; larger industrial refrigeration coils are listed much higher. (TradeIndia listings.)
- Refrigerator door gaskets: public listings include roughly ₹90–₹175 per metre for common gasket profiles. (TradeIndia listings.)
- Refrigeration copper tube: approximately ₹850/kg in a Mumbai listing. (TradeIndia.)
- Copper capillary tube: approximately ₹590/kg in a Mumbai listing. (TradeIndia.)
- Refrigerator starter relay: approximately ₹33/piece. (TradeIndia.)
- Refrigerator controller: public listed range approximately ₹600–₹15,000 depending on controller specification. (TradeIndia.)
- Refrigerator PCB boards: public listings commonly show approximately ₹1,500–₹3,100/piece depending on model and board type. (TradeIndia.)
- Refrigerator fan motors: public listings include approximately ₹450/piece and a ₹1,000–₹1,500/piece range for certain fan-motor configurations. (TradeIndia.)
- Refrigerator defrost heaters: public listings include approximately ₹250–₹550/piece, with one listed refrigerator defrost heater around ₹500. (TradeIndia.)
- R600a refrigerant: public listings vary by pack/unit; examples include about ₹160 for a 200 g can and approximately ₹480–₹650/kg for bulk/cylinder offerings. (TradeIndia / other Indian B2B listings.)

These references are **benchmark inputs, not exact supplier quotes**. Actual OEM pricing varies materially by annual volume, specification, certification, freight, tooling, payment terms, and contract conditions.

## Supplier quote logic

For each SKU, three synthetic suppliers are generated with:
- small price dispersion around the benchmark anchor,
- different on-time delivery performance,
- different quality performance,
- different lead times.

The lowest-price supplier is therefore not always the highest-scoring supplier. This is intentional so ProcuraX can demonstrate a genuine procurement trade-off between cost, quality, delivery and lead time.


## v119 multi-category demo data
The original Refrigerator Category history and supplier records are retained unchanged. Two additional synthetic portfolios were generated for the hackathon demo: Air-Conditioning Category and Washing Machine Category. Each contains 66 SKUs (33 component families × 2 variants), daily historical demand from 1 Jan 2025 to 10 Sep 2026, inventory/current-stock inputs, UOM, and three supplier records per SKU. Demand profiles include category-appropriate seasonal patterns and modest trend/noise so the forecasting and procurement decision engine produces varied outcomes. Supplier quotes are synthetic demo inputs designed to create realistic competitive spreads; they are not represented as actual supplier contracts.
