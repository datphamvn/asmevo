# GCC baseline check (T7)

Mục tiêu: kiểm tra `baseline_asm` có sẵn trong dataset (mốc tính mọi speedup) có khớp với
`gcc -O3 -S` của toolchain mình hay không, và nếu lệch thì lệch bao nhiêu. Kết quả dùng cho
phần *threats to validity*.

## Cách chạy

```bash
uv run asmevo gcc-baseline --limit 5 --cc gcc-12   # hoặc --ids 0 3 7, --out runs/<dir>
uv run pytest tests/test_gcc_baseline.py
```

Với mỗi problem, `src/asmevo/eval/gcc_baseline.py`:

1. Build `baseline_asm` của dataset; bỏ test mà nó vượt `run_timeout_s` (giống `evaluate`).
2. Compile `c_code` bằng `<cc> -O3 -S temp.c`, rồi build bằng `eval.cc`/`link_flags` như candidate.
3. So asm sau khi chuẩn hoá (bỏ comment, `.file`, `.ident`, khoảng trắng): `identical` và
   `diff_lines` (số dòng thêm + bớt).
4. Chạy toàn bộ test trên cả hai; nếu cả hai pass thì đo bằng hyperfine (3 warmup, 10 lần đo
   mỗi test, hai binary trong cùng một lệnh). `ratio = T_ours / T_dataset` (> 1 là bản của mình
   chậm hơn).

Output: `comparison.jsonl` (một dòng một problem) và `summary.json` (geomean/min/max `ratio`).

## Môi trường

| | Compiler |
|---|---|
| Dataset (`.ident` trong `baseline_asm`) | GCC 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04) |
| Docker image (`docker/Dockerfile`) | gcc-12 của Ubuntu 22.04 |
| Lần chạy này (host Ubuntu 26.04, x86-64) | gcc-11 11.5.0-11ubuntu2, gcc-12 12.5.0-9ubuntu1, hyperfine |

Ngày chạy: 05/10/2026. Problem `0`–`4` của split `val` (103–104 test mỗi problem, không test nào
bị drop).

## Kết quả

| Problem | gcc-11: identical / diff | gcc-11: ratio | gcc-12: identical / diff | gcc-12: ratio | T_dataset (s, tổng các test) |
|---|---|---|---|---|---|
| 0 | ✅ / 0 | 0.997 | ❌ / 20 | 0.998 | 0.216 |
| 1 | ❌ / 4 | 0.997 | ❌ / 6 | 1.001 | 0.254 |
| 2 | ✅ / 0 | 0.998 | ❌ / 2 | 1.002 | 0.218 |
| 3 | ✅ / 0 | 1.001 | ❌ / 16 | 0.998 | 0.217 |
| 4 | ✅ / 0 | 1.000 | ❌ / 117 | 1.003 | 0.217 |
| **Tổng** | 4/5 identical | geomean **0.999** | 0/5 identical | geomean **1.000** | |

Cả hai compiler: 5/5 compile được, 5/5 pass toàn bộ test.

Các khác biệt quan sát được:
- **gcc-11, problem 1:** chỉ khác số thứ tự label `.LFB35`/`.LFE35` → `.LFB38`/`.LFE38` (header
  glibc mới hơn khai báo thêm hàm). Không đổi lệnh nào.
- **gcc-12, problem 1:** ngoài label, `movl $1, %edi` → `movl $2, %edi` trước `__printf_chk`, tức
  mức `_FORTIFY_SOURCE` mặc định của gcc Ubuntu mới khác với của dataset.
- **gcc-12, problem 4:** vòng lặp vector hoá SSE được cấp phát thanh ghi và sắp xếp lại khác
  (117 dòng khác), nhưng thời gian vẫn như nhau.

## Kết luận và threats to validity

1. **Giữ `baseline_asm` của dataset làm mốc.** gcc-11 tái tạo được gần như y hệt (4/5 trùng khớp,
   1/5 chỉ khác label), và cả gcc-11 lẫn gcc-12 đều cho thời gian lệch không quá ±0.35%. Không cần build lại
   baseline bằng toolchain trong Docker.
2. **Toolchain không khớp hoàn toàn.** Dataset dùng gcc 11.4 (Ubuntu 22.04), còn Docker dùng
   gcc-12. Asm của gcc-12 khác ở mọi problem đã thử: cấp phát thanh ghi, vector hoá, và các mặc
   định theo distro như `_FORTIFY_SOURCE`. Vì vậy speedup được đo so với *gcc 11.4 -O3*, không phải
   gcc -O3 mới nhất. Cần nêu rõ điều này trong báo cáo.
3. **Ngưỡng nhiễu.** Ngay cả khi hai binary giống hệt nhau (gcc-11, problem 0/2/3/4), ratio vẫn
   dao động trong khoảng 0.997–1.001. Speedup dưới khoảng **1%** không nên coi là cải thiện thật.
   T13 có thể dùng con số này làm ngưỡng tối thiểu cho final check.
4. **Thời gian chủ yếu là khởi động process.** Khoảng 0.22 s cho ~104 test, tức ~2 ms mỗi lần
   chạy. Với các problem như vậy, thời gian khởi động process chiếm phần lớn, nên tối ưu asm khó
   tạo ra speedup lớn. Đây là giới hạn của benchmark khi diễn giải RQ1.
5. **Phạm vi nhỏ.** Mới kiểm tra 5/200 problem trên một máy host (không phải trong Docker). Muốn mở
   rộng thì chạy `--limit 200`.
