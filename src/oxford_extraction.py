import polars as pl
import urllib.request
import zipfile
import os

def instantiate_oxford_man_archive():
    if not os.path.exists("data"):
        os.makedirs("data")
        
    url = "https://github.com/onnokleen/mfGARCH/raw/v0.1.9/data-raw/OxfordManRealizedVolatilityIndices.zip"
    zip_path = "data/OxfordMan.zip"
    csv_path = "data/OxfordManRealizedVolatilityIndices.csv"
    
    if not os.path.exists(csv_path):
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data")
            
    df = pl.read_csv(csv_path)
    
    spx_rv = (
        df.filter(pl.col("Symbol") == ".SPX")
        .select([
            pl.col("").str.slice(0, 10).alias("Date"),
            pl.col("rv5").alias("Realized_Variance")
        ])
        .drop_nulls()
        .with_columns(
            Log_Variance = pl.col("Realized_Variance").log()
        )
    )
    
    export_path = "data/SPX_Realized_Variance_Oxford.xlsx"
    spx_rv.write_excel(export_path)
    print(f"Empirical Matrix Serialized: {export_path}")

if __name__ == "__main__":
    instantiate_oxford_man_archive()