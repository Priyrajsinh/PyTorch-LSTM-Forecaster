"""Pandera schema for Jena Climate dataset validation."""
import pandera as pa
from pandera import Column, DataFrameSchema

CLIMATE_SCHEMA = DataFrameSchema(
    {
        "p (mbar)": Column(float),
        "T (degC)": Column(float),
        "Tpot (K)": Column(float),
        "Tdew (degC)": Column(float),
        "rh (%)": Column(float),
        "VPmax (mbar)": Column(float),
        "VPact (mbar)": Column(float),
        "VPdef (mbar)": Column(float),
        "sh (g/kg)": Column(float),
        "H2OC (mmol/mol)": Column(float),
        "rho (g/m**3)": Column(float),
        "wv (m/s)": Column(float),
        "max. wv (m/s)": Column(float),
        "wd_sin": Column(float),
        "wd_cos": Column(float),
    },
    strict=False,
)
