# GCC baseline check (T7)

Mục tiêu: kiểm tra `baseline_asm` có sẵn trong dataset (mốc tính mọi speedup) có khớp với
`gcc -O3 -S` của toolchain mình hay không, và nếu lệch thì lệch bao nhiêu. Kết quả dùng cho
phần *threats to validity*.

## Cách chạy

Trong Docker (toolchain chuẩn, `gcc` = 11.4.0 như dataset):

```bash
docker build -f docker/Dockerfile -t asmevo .
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/work asmevo \
  bash -c "uv run --no-sync pytest -q && uv run --no-sync asmevo gcc-baseline --limit 5"
```

Trên host (cần gcc và hyperfine; `--cc` để chọn compiler khác `eval.cc`):

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
| Docker image (`docker/Dockerfile`, Ubuntu 22.04) | GCC 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04.3), hyperfine 1.20.0 |
| Host (Ubuntu 26.04, x86-64), để so sánh thêm | gcc-11 11.5.0-11ubuntu2, gcc-12 12.5.0-9ubuntu1 |

Ban đầu Dockerfile cài gcc-12. Lần chạy đầu trên host phát hiện dataset dùng gcc 11.4, nên
Dockerfile đã được đổi sang `gcc-11` (gcc mặc định của Ubuntu 22.04). Lúc build image, nếu
`gcc -dumpfullversion` khác `11.4.0` thì build báo lỗi. Bản trong image chỉ khác dataset ở số
revision đóng gói của Ubuntu (`~22.04.3` so với `~22.04`); phiên bản upstream vẫn là 11.4.0.

Problem `0`–`4` của split `val` (103–104 test mỗi problem, không test nào bị drop). Host chạy
ngày 05/10/2026, Docker chạy ngày 06/10/2026.

## Kết quả

### Docker: gcc 11.4.0, toolchain chính thức

| Problem | identical / diff | T_dataset (s) | T_ours (s) | ratio |
|---|---|---|---|---|
| 0 | ✅ / 0 | 0.186 | 0.185 | 0.998 |
| 1 | ✅ / 0 | 0.224 | 0.224 | 1.003 |
| 2 | ✅ / 0 | 0.188 | 0.188 | 0.999 |
| 3 | ✅ / 0 | 0.183 | 0.182 | 0.992 |
| 4 | ✅ / 0 | 0.185 | 0.185 | 1.000 |
| **Tổng** | **5/5 identical** | | | geomean **0.998** |

5/5 compile được, 5/5 pass toàn bộ test, và asm giống hệt dataset, kể cả số thứ tự label.

### Host: gcc-11.5 và gcc-12.5, để thấy ảnh hưởng của lệch version

| Problem | gcc-11: identical / diff | gcc-11: ratio | gcc-12: identical / diff | gcc-12: ratio | T_dataset (s, tổng các test) |
|---|---|---|---|---|---|
| 0 | ✅ / 0 | 0.997 | ❌ / 20 | 0.998 | 0.216 |
| 1 | ❌ / 4 | 0.997 | ❌ / 6 | 1.001 | 0.254 |
| 2 | ✅ / 0 | 0.998 | ❌ / 2 | 1.002 | 0.218 |
| 3 | ✅ / 0 | 1.001 | ❌ / 16 | 0.998 | 0.217 |
| 4 | ✅ / 0 | 1.000 | ❌ / 117 | 1.003 | 0.217 |
| **Tổng** | 4/5 identical | geomean **0.999** | 0/5 identical | geomean **1.000** | |

Cả hai compiler trên host: 5/5 compile được, 5/5 pass toàn bộ test.

Các khác biệt quan sát được:
- **gcc-11, problem 1:** chỉ khác số thứ tự label `.LFB35`/`.LFE35` → `.LFB38`/`.LFE38` (header
  glibc mới hơn khai báo thêm hàm). Không đổi lệnh nào.
- **gcc-12, problem 1:** ngoài label, `movl $1, %edi` → `movl $2, %edi` trước `__printf_chk`, tức
  mức `_FORTIFY_SOURCE` mặc định của gcc Ubuntu mới khác với của dataset.
- **gcc-12, problem 4:** vòng lặp vector hoá SSE được cấp phát thanh ghi và sắp xếp lại khác
  (117 dòng khác), nhưng thời gian vẫn như nhau.

## Kết luận và threats to validity

1. **Baseline tái tạo được hoàn toàn.** Với toolchain Docker (gcc 11.4.0), `gcc -O3 -S` cho asm
   giống hệt `baseline_asm` ở 5/5 problem. Vì vậy dùng `baseline_asm` làm mốc 1.0× tương đương với
   tự compile bằng toolchain của nhóm.
2. **Version compiler đã được khớp.** Lệch version (dataset gcc 11.4, Docker cũ gcc-12) đã được
   sửa trong Dockerfile. Speedup báo cáo là so với *gcc 11.4 -O3* (Ubuntu 22.04), không phải gcc
   mới nhất. Trên host, gcc-12 sinh asm khác ở mọi problem (cấp phát thanh ghi, vector hoá,
   `_FORTIFY_SOURCE`) nhưng thời gian chỉ lệch ≤ 0.3%, nên kết luận ít phụ thuộc vào version.
3. **Ngưỡng nhiễu.** Ở 9 cặp binary giống hệt nhau (Docker 5/5 và gcc-11 trên host 4/5), ratio vẫn
   dao động 0.992–1.003, tức tới khoảng ±0.8%. Speedup dưới khoảng **1%** không nên coi là cải
   thiện thật. T13 có thể dùng con số này làm ngưỡng tối thiểu cho final check.
4. **Thời gian chủ yếu là khởi động process.** Khoảng 0.18–0.25 s cho ~104 test, tức ~2 ms mỗi lần
   chạy. Với các problem như vậy, thời gian khởi động process chiếm phần lớn, nên tối ưu asm khó
   tạo ra speedup lớn. Đây là giới hạn của benchmark khi diễn giải RQ1.
5. **Phạm vi nhỏ.** Mới kiểm tra 5/200 problem trên một máy. Muốn mở rộng thì chạy `--limit 200`.
