from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from notion_client import Client
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pydantic import BaseModel
import asyncio
import time

# Load environment variables
load_dotenv()

app = FastAPI()

# タイムアウト設定
app.timeout = 60  # 60秒

# キャッシュ設定
cache = {}
CACHE_DURATION = 300  # 5分

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Notion client setup
notion = Client(auth=os.getenv("NOTION_TOKEN"))
database_id = os.getenv("NOTION_DATABASE_ID")

class StockPosition(BaseModel):
    quantity: float
    price: float

class Stock(BaseModel):
    ticker: str
    positions: List[StockPosition]

def get_exchange_rate():
    """USD/JPYの為替レートを取得"""
    try:
        usdjpy = yf.Ticker("USDJPY=X")
        return usdjpy.history(period="1d")['Close'].iloc[-1]
    except Exception as e:
        return None

def calculate_fair_value(ticker: str):
    """
    マルチプル法とDDMを使用して適正株価を計算
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 基本データの取得
        current_price = info.get('currentPrice', 0)
        if not current_price:
            return None
            
        # PERによる評価
        eps = info.get('trailingEps', 0)
        sector = info.get('sector', '')
        
        # セクター平均PERの取得（実際はもっと精緻な実装が必要）
        sector_per = 15  # デフォルト値
        
        per_fair_value = eps * sector_per if eps else 0
        
        # EV/EBITDA評価
        ebitda = info.get('ebitda', 0)
        enterprise_value = info.get('enterpriseValue', 0)
        if ebitda and enterprise_value:
            ev_ebitda = enterprise_value / ebitda
            sector_ev_ebitda = 10  # デフォルト値
            ev_ebitda_fair_value = (ebitda * sector_ev_ebitda) / info.get('sharesOutstanding', 1)
        else:
            ev_ebitda_fair_value = 0
        
        # DDMによる評価
        dividend_rate = info.get('dividendRate', 0)
        if dividend_rate:
            growth_rate = 0.02  # 2%の成長率を仮定
            cost_of_equity = 0.08  # 8%の株主資本コストを仮定
            ddm_fair_value = dividend_rate * (1 + growth_rate) / (cost_of_equity - growth_rate)
        else:
            ddm_fair_value = 0
        
        # 各手法の重み付け
        weights = {
            'per': 0.4,
            'ev_ebitda': 0.4,
            'ddm': 0.2
        }
        
        fair_value = (
            per_fair_value * weights['per'] +
            ev_ebitda_fair_value * weights['ev_ebitda'] +
            ddm_fair_value * weights['ddm']
        )
        
        return {
            'fair_value': fair_value,
            'current_price': current_price,
            'deviation_rate': ((current_price - fair_value) / fair_value) * 100 if fair_value else 0
        }
    except Exception as e:
        return None

def get_cached_data(key: str):
    if key in cache:
        data, timestamp = cache[key]
        if time.time() - timestamp < CACHE_DURATION:
            return data
    return None

def set_cached_data(key: str, data: any):
    cache[key] = (data, time.time())

async def get_stock_info_with_timeout(ticker: str):
    try:
        async with asyncio.timeout(30):  # 30秒タイムアウト
            cached_data = get_cached_data(ticker)
            if cached_data:
                return cached_data
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # 必要な情報のみを取得して処理を軽量化
            result = {
                "name": info.get("longName", ""),
                "current_price": info.get("currentPrice", 0),
                "currency": "JPY" if ticker.endswith(".T") else "USD",
            }
            
            set_cached_data(ticker, result)
            return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="データ取得がタイムアウトしました")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/exchange-rate")
async def get_current_exchange_rate():
    rate = get_exchange_rate()
    if rate:
        return {"rate": rate}
    raise HTTPException(status_code=500, detail="為替レートの取得に失敗しました")

@app.get("/api/stock/{ticker}")
async def get_stock_info(ticker: str):
    try:
        stock_info = await get_stock_info_with_timeout(ticker)
        if not stock_info:
            raise HTTPException(status_code=500, detail="株価情報の取得に失敗しました")
        
        # 適正株価の計算
        fair_value_info = calculate_fair_value(ticker)
        if not fair_value_info:
            raise HTTPException(status_code=500, detail="適正株価の計算に失敗しました")
        
        # 為替レートの取得（米国株の場合）
        exchange_rate = None
        if not ticker.endswith(".T"):  # 日本株以外
            exchange_rate = get_exchange_rate()
        
        return {
            "name": stock_info["name"],
            "current_price": stock_info["current_price"],
            "currency": stock_info["currency"],
            "exchange_rate": exchange_rate,
            "fair_value": fair_value_info["fair_value"],
            "deviation_rate": fair_value_info["deviation_rate"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stocks")
async def add_stock(stock: Stock):
    try:
        # Notionデータベースに保存
        notion.pages.create(
            parent={"database_id": database_id},
            properties={
                "Ticker": {"title": [{"text": {"content": stock.ticker}}]},
                "Positions": {"rich_text": [{"text": {"content": str([{"quantity": p.quantity, "price": p.price} for p in stock.positions])}}]},
                "Average Price": {"number": sum(p.price * p.quantity for p in stock.positions) / sum(p.quantity for p in stock.positions)},
                "Total Quantity": {"number": sum(p.quantity for p in stock.positions)}
            }
        )
        return {"message": "Stock position added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stocks")
async def get_stocks():
    try:
        response = notion.databases.query(database_id=database_id)
        stocks = []
        for page in response["results"]:
            ticker = page["properties"]["Ticker"]["title"][0]["text"]["content"]
            positions = eval(page["properties"]["Positions"]["rich_text"][0]["text"]["content"])
            average_price = page["properties"]["Average Price"]["number"]
            total_quantity = page["properties"]["Total Quantity"]["number"]
            
            # 現在の株価情報を取得
            stock_info = await get_stock_info_with_timeout(ticker)
            
            stocks.append({
                "ticker": ticker,
                "positions": positions,
                "average_price": average_price,
                "total_quantity": total_quantity,
                "current_info": stock_info
            })
        return stocks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
