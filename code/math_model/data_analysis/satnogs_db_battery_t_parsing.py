#!/usr/bin/env python3
"""
Выгрузка и парсинг температуры АКБ из телеметрии SatNOGS DB для нескольких МКА.

Требования:  pip install requests pandas matplotlib
Токен:       https://db.satnogs.org -> профиль -> API Token
             export SATNOGS_TOKEN=token

Примеры:
  # 1) Посмотреть, какие поля вообще есть в декодированной телеметрии спутника
  python satnogs_battery_temp.py --discover 43880

  # 2) Скачать температуру АКБ за период для спутников из SATELLITES
  python satnogs_battery_temp.py --start 2026-09-01 --end 2026-10-25 --plot

  # 3) Свои спутники и своё регулярное выражение для имени поля
  python satnogs_battery_temp.py --norad 43880 47959 43802 \
      --regex "(batt|bat|акб).*temp|temp.*(batt|bat)" --start 2026-10-01
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

API = "https://db.satnogs.org/api/telemetry/"

# Примеры КА (NORAD ID -> имя). ПРОВЕРЬТЕ на db.satnogs.org, что у спутника
# есть декодер и в телеметрии присутствует температура АКБ.
SATELLITES = {
    43880: "UWE-4",
    47959: "GRBAlpha",
    43802: "Suomi 100",
}

# Имена полей у разных КА различаются (decoders на Kaitai), поэтому ищем по regex.
# Можно задать точное имя поля для конкретного КА в FIELD_OVERRIDES.
DEFAULT_REGEX = r"(batt|bat_|battery|акб).*(temp|_t$|tmp)|(temp|tmp).*(batt|bat_|battery)"
FIELD_OVERRIDES = {
    # 43880: "bat_temp",   # пример
}


def session():
    token = os.environ.get("SATNOGS_TOKEN")
    s = requests.Session()
    if token:
        s.headers["Authorization"] = f"Token {token}"
    else:
        print("[!] SATNOGS_TOKEN не задан: API телеметрии, скорее всего, вернёт 401/403",
              file=sys.stderr)
    s.headers["Accept"] = "application/json"
    return s


def get_with_retry(s, url, params, timeout, attempts=4):
    """GET с повтором при таймауте/сетевой ошибке (экспоненциальная пауза)."""
    for i in range(attempts):
        try:
            return s.get(url, params=params, timeout=(15, timeout))
        except (requests.ReadTimeout, requests.ConnectTimeout, requests.ConnectionError) as e:
            wait = 5 * 2 ** i
            print(f"    {type(e).__name__}, повтор через {wait} с [{i + 1}/{attempts}]",
                  file=sys.stderr)
            time.sleep(wait)
    raise requests.Timeout("db.satnogs.org не отвечает после нескольких попыток")


def parse_date(text):
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def fetch_windowed(s, norad, start, end, window_days=7, timeout=180):
    """Разбивает период на окна, чтобы запросы были короткими.
    Если start не задан - берутся последние 30 суток."""
    t_end = parse_date(end) if end else datetime.now(timezone.utc)
    t_start = parse_date(start) if start else t_end - timedelta(days=30)
    cur = t_start
    while cur < t_end:
        nxt = min(cur + timedelta(days=window_days), t_end)
        print(f"    окно {cur:%Y-%m-%d} .. {nxt:%Y-%m-%d}")
        yield from fetch_frames(s, norad, cur.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                nxt.strftime("%Y-%m-%dT%H:%M:%SZ"), timeout=timeout)
        cur = nxt


def fetch_frames(s, norad, start=None, end=None, max_pages=200, pause=0.5, timeout=180):
    """Постраничная загрузка кадров телеметрии (cursor-пагинация)."""
    params = {"satellite": norad, "format": "json"}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    url, page = API, 0
    while url and page < max_pages:
        r = get_with_retry(s, url, params if page == 0 else None, timeout)
        if r.status_code == 429:  # лимит запросов
            time.sleep(int(r.headers.get("Retry-After", 10)))
            continue
        r.raise_for_status()
        data = r.json()
        # Ответ может быть списком (next в заголовке Link) или dict с "results"/"next"
        if isinstance(data, dict):
            rows, nxt = data.get("results", []), data.get("next")
        else:
            rows, nxt = data, r.links.get("next", {}).get("url")
        yield from rows
        url, page = nxt, page + 1
        time.sleep(pause)


def flatten(obj, prefix=""):
    """Разворачивает вложенный JSON декодера в плоский словарь 'a.b.c' -> значение."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def parse_decoded(frame):
    d = frame.get("decoded")
    if not d:
        return {}
    if isinstance(d, str):
        try:
            d = json.loads(d)
        except json.JSONDecodeError:
            return {}
    return flatten(d)


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def extract_temperature(frames, norad, regex):
    """Из списка кадров -> DataFrame [timestamp, field, value, observer]."""
    field_re = re.compile(regex, re.I)
    forced = FIELD_OVERRIDES.get(norad)
    rows = []
    for f in frames:
        flat = parse_decoded(f)
        if not flat:
            continue
        for key, val in flat.items():
            hit = (key.split(".")[-1] == forced or key == forced) if forced \
                else field_re.search(key)
            num = to_float(val)
            if hit and num is not None:
                rows.append({
                    "timestamp": f.get("timestamp"),
                    "field": key,
                    "temp_c": num,
                    "observer": f.get("observer"),
                })
    if not rows:
        return pd.DataFrame(columns=["timestamp", "field", "temp_c", "observer"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(
        ["timestamp", "field"]).reset_index(drop=True)


def discover(s, norad, n=200):
    """Показывает имена числовых полей в последних n кадрах — чтобы найти поле температуры АКБ."""
    keys = {}
    for i, f in enumerate(fetch_windowed(s, norad, None, None, window_days=3)):
        for k, v in parse_decoded(f).items():
            if to_float(v) is not None:
                keys.setdefault(k, v)
        if i >= n:
            break
    if not keys:
        print("Декодированных кадров нет (у КА может не быть декодера или нет доступа).")
    for k in sorted(keys):
        mark = "  <-- кандидат" if re.search(DEFAULT_REGEX, k, re.I) else ""
        print(f"{k:60s} пример: {keys[k]}{mark}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--norad", type=int, nargs="+", help="NORAD ID (по умолчанию SATELLITES)")
    ap.add_argument("--start", help="начало периода, напр. 2026-09-01")
    ap.add_argument("--end", help="конец периода")
    ap.add_argument("--regex", default=DEFAULT_REGEX, help="regex имени поля температуры АКБ")
    ap.add_argument("--discover", type=int, metavar="NORAD", help="вывести доступные поля КА")
    ap.add_argument("--window-days", type=int, default=7, help="размер окна запроса, суток")
    ap.add_argument("--timeout", type=int, default=180, help="таймаут чтения ответа, с")
    ap.add_argument("--out", default="telemetry_out", help="папка для CSV/графиков")
    ap.add_argument("--plot", action="store_true", help="построить графики")
    args = ap.parse_args()

    s = session()
    if args.discover:
        try:
            discover(s, args.discover)
        except requests.RequestException as e:
            print(f"Ошибка API: {e}", file=sys.stderr)
        return

    sats = {n: SATELLITES.get(n, str(n)) for n in (args.norad or SATELLITES)}
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    frames_all = []
    for norad, name in sats.items():
        print(f"[*] {name} ({norad}): загрузка...")
        try:
            frames = list(fetch_windowed(s, norad, args.start, args.end,
                                         args.window_days, args.timeout))
        except requests.RequestException as e:
            print(f"    ошибка API: {e}", file=sys.stderr)
            continue
        df = extract_temperature(frames, norad, args.regex)
        print(f"    кадров: {len(frames)}, точек температуры АКБ: {len(df)}"
              + (f", поля: {sorted(df['field'].unique())}" if len(df) else ""))
        if df.empty:
            continue
        df.insert(0, "satellite", name)
        df.insert(1, "norad", norad)
        safe_name = re.sub(r"\W+", "_", name)
        df.to_csv(out / f"{norad}_{safe_name}_battery_temp.csv", index=False)
        frames_all.append(df)

    if not frames_all:
        print("Данных не найдено. Запустите --discover NORAD и уточните --regex.")
        return

    all_df = pd.concat(frames_all, ignore_index=True)
    all_df.to_csv(out / "all_battery_temp.csv", index=False)

    # Краткая статистика для тепловой оценки
    stats = all_df.groupby("satellite")["temp_c"].agg(
        ["count", "min", "mean", "max", "std"]).round(2)
    print("\n", stats)
    stats.to_csv(out / "stats.csv")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, g in all_df.groupby("satellite"):
            ax.plot(g["timestamp"], g["temp_c"], ".", ms=3, label=name)
        ax.set_ylabel("Температура АКБ, °C")
        ax.set_xlabel("Время (UTC)")
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "battery_temp.png", dpi=150)
        print(f"График: {out / 'battery_temp.png'}")


if __name__ == "__main__":
    main()