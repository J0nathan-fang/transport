"""
/api/predict 接口并发测试脚本
用于论文：单节点环境下端到端平均延迟测试
"""

import asyncio
import json
import statistics
import time
import httpx
import argparse

API_URL = "http://localhost:8081/api/predict"
TEST_PAYLOAD = {"lng": 103.987, "lat": 30.761}


async def single_request(client: httpx.AsyncClient, request_id: int) -> dict:
    start = time.perf_counter()
    try:
        resp = await client.post(API_URL, json=TEST_PAYLOAD)
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "id": request_id,
            "status": resp.status_code,
            "latency_ms": round(elapsed, 2),
            "success": resp.status_code == 200,
        }
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "id": request_id,
            "status": 0,
            "latency_ms": round(elapsed, 2),
            "success": False,
            "error": str(e),
        }


async def run_concurrent_test(concurrency: int, total_requests: int):
    print(f"\n{'='*60}")
    print(f"  /api/predict 并发测试")
    print(f"{'='*60}")
    print(f"  目标地址:     {API_URL}")
    print(f"  并发数:       {concurrency}")
    print(f"  总请求数:     {total_requests}")
    print(f"  测试载荷:     {json.dumps(TEST_PAYLOAD, ensure_ascii=False)}")
    print(f"{'='*60}\n")

    results = []
    wall_start = time.perf_counter()

    async with httpx.AsyncClient(timeout=30.0) as client:
        sem = asyncio.Semaphore(concurrency)

        async def bounded_request(req_id):
            async with sem:
                return await single_request(client, req_id)

        tasks = [bounded_request(i) for i in range(total_requests)]
        results = await asyncio.gather(*tasks)

    wall_elapsed = (time.perf_counter() - wall_start) * 1000

    success_results = [r for r in results if r["success"]]
    fail_results = [r for r in results if not r["success"]]
    latencies = [r["latency_ms"] for r in success_results]

    print(f"{'='*60}")
    print(f"  测试结果")
    print(f"{'='*60}")
    print(f"  总请求数:       {total_requests}")
    print(f"  成功:           {len(success_results)}")
    print(f"  失败:           {len(fail_results)}")
    print(f"  成功率:         {len(success_results)/total_requests*100:.1f}%")
    print(f"  总耗时(墙钟):   {wall_elapsed:.0f} ms")
    print(f"  吞吐量(QPS):    {total_requests/(wall_elapsed/1000):.2f} req/s")
    print(f"{'─'*60}")
    print(f"  延迟统计 (端到端):")
    print(f"{'─'*60}")

    if latencies:
        latencies_sorted = sorted(latencies)
        print(f"  平均延迟:       {statistics.mean(latencies):.2f} ms")
        print(f"  中位数(P50):    {statistics.median(latencies):.2f} ms")
        print(f"  P90:            {latencies_sorted[int(len(latencies_sorted)*0.9)]:.2f} ms")
        print(f"  P95:            {latencies_sorted[int(len(latencies_sorted)*0.95)]:.2f} ms")
        print(f"  P99:            {latencies_sorted[int(len(latencies_sorted)*0.99)]:.2f} ms")
        print(f"  最小延迟:       {min(latencies):.2f} ms")
        print(f"  最大延迟:       {max(latencies):.2f} ms")
        print(f"  标准差:         {statistics.stdev(latencies):.2f} ms")

    if fail_results:
        print(f"\n  失败请求详情:")
        for r in fail_results[:5]:
            print(f"    #{r['id']}: status={r['status']}, error={r.get('error','N/A')}")

    print(f"{'='*60}\n")

    with open("benchmark_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "url": API_URL,
                "concurrency": concurrency,
                "total_requests": total_requests,
                "payload": TEST_PAYLOAD,
            },
            "summary": {
                "total": total_requests,
                "success": len(success_results),
                "failed": len(fail_results),
                "success_rate": f"{len(success_results)/total_requests*100:.1f}%",
                "wall_time_ms": round(wall_elapsed, 2),
                "qps": round(total_requests / (wall_elapsed / 1000), 2),
                "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
                "p50_ms": round(statistics.median(latencies), 2) if latencies else None,
                "p90_ms": round(latencies_sorted[int(len(latencies_sorted)*0.9)], 2) if latencies else None,
                "p95_ms": round(latencies_sorted[int(len(latencies_sorted)*0.95)], 2) if latencies else None,
                "p99_ms": round(latencies_sorted[int(len(latencies_sorted)*0.99)], 2) if latencies else None,
                "min_ms": round(min(latencies), 2) if latencies else None,
                "max_ms": round(max(latencies), 2) if latencies else None,
                "stddev_ms": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else None,
            },
            "details": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"  详细结果已保存至: benchmark_result.json")


async def run_multi_concurrency_test(total_requests: int):
    concurrency_levels = [1, 5, 10, 20, 50]
    print(f"\n{'#'*60}")
    print(f"  多并发级别对比测试")
    print(f"{'#'*60}")

    summary_table = []

    for c in concurrency_levels:
        print(f"\n>>> 并发数 = {c}")
        results = []
        wall_start = time.perf_counter()

        async with httpx.AsyncClient(timeout=30.0) as client:
            sem = asyncio.Semaphore(c)

            async def bounded_request(req_id):
                async with sem:
                    return await single_request(client, req_id)

            tasks = [bounded_request(i) for i in range(total_requests)]
            results = await asyncio.gather(*tasks)

        wall_elapsed = (time.perf_counter() - wall_start) * 1000
        latencies = [r["latency_ms"] for r in results if r["success"]]
        success_count = len(latencies)

        if latencies:
            latencies_sorted = sorted(latencies)
            row = {
                "concurrency": c,
                "avg_ms": round(statistics.mean(latencies), 2),
                "p50_ms": round(statistics.median(latencies), 2),
                "p90_ms": round(latencies_sorted[int(len(latencies_sorted)*0.9)], 2),
                "p95_ms": round(latencies_sorted[int(len(latencies_sorted)*0.95)], 2),
                "qps": round(total_requests / (wall_elapsed / 1000), 2),
                "success_rate": f"{success_count/total_requests*100:.1f}%",
            }
            summary_table.append(row)

    print(f"\n{'='*80}")
    print(f"  对比汇总表 (总请求数={total_requests})")
    print(f"{'='*80}")
    print(f"  {'并发数':>6} | {'平均延迟':>10} | {'P50':>10} | {'P90':>10} | {'P95':>10} | {'QPS':>8} | {'成功率':>8}")
    print(f"  {'─'*6}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*8}")
    for row in summary_table:
        print(f"  {row['concurrency']:>6} | {row['avg_ms']:>8.2f}ms | {row['p50_ms']:>8.2f}ms | {row['p90_ms']:>8.2f}ms | {row['p95_ms']:>8.2f}ms | {row['qps']:>6.2f} | {row['success_rate']:>8}")
    print(f"{'='*80}\n")

    with open("benchmark_multi_result.json", "w", encoding="utf-8") as f:
        json.dump(summary_table, f, ensure_ascii=False, indent=2)
    print(f"  对比结果已保存至: benchmark_multi_result.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="/api/predict 并发测试")
    parser.add_argument("-c", "--concurrency", type=int, default=10, help="并发数 (默认10)")
    parser.add_argument("-n", "--total", type=int, default=100, help="总请求数 (默认100)")
    parser.add_argument("--multi", action="store_true", help="运行多并发级别对比测试")
    args = parser.parse_args()

    if args.multi:
        asyncio.run(run_multi_concurrency_test(args.total))
    else:
        asyncio.run(run_concurrent_test(args.concurrency, args.total))
