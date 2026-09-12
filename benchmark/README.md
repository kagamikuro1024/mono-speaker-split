# Benchmark

Chạy cả bộ mẫu qua `monosplit.separate`, chấm từng ca so với kỳ vọng ghi trong
`cases.json`, in bảng markdown ra màn hình và ghi `benchmark/report.json`.

Không có mẫu giả ở đây: mô hình thật, tệp thật, đồng hồ thật.

## Chạy

```bash
MONOSPLIT_MODELS=$HOME/.cache/voice-diar \
    uv run --with-editable '.[asr]' python -m benchmark.run
```

Tuỳ chọn:

| Cờ | Ý nghĩa |
|---|---|
| `--audio DIR` | Lấy mẫu trong `DIR` thay cho đường dẫn ghi trong `cases.json` (khớp theo **tên tệp**). |
| `--cases FILE` | Dùng bộ ca khác. Mặc định `benchmark/cases.json`. |
| `--no-asr` | Bỏ chép lời. Nhanh hơn nhiều, nhưng lớp 2 mất căn cứ chữ nên phải đoán vai theo thứ tự lượt. |
| `--models DIR` | Thư mục chứa `seg.onnx` / `emb.onnx`. Không truyền thì dùng `MONOSPLIT_MODELS`, không có nữa thì tải về `~/.cache/monosplit`. |
| `--out FILE` | Nơi ghi báo cáo JSON. Mặc định `benchmark/report.json`. |

Mã thoát `0` khi mọi ca đều đạt, `1` khi có ca trượt — đủ để cắm vào CI nếu bạn
có bộ mẫu cố định.

## Đọc bảng

| Cột | Nghĩa |
|---|---|
| **Ca** | Tên tệp. |
| **Chế độ** | `Result.mode`: `mono`, `stereo_trung_nhau` (hai kênh trùng nhau — mono nhân đôi), `stereo_mot_ben_cam` (một kênh câm). |
| **Giây** | Thời gian tường chạy ca đó, kể cả giải mã và chép lời. |
| **Lượt** | Số lượt nói trong kết quả. |
| **Khách/Agent** | Số lượt gán cho từng vai. Lệch hẳn về một bên là dấu hiệu lớp 1 gom hụt. |
| **Margin** | Trung vị khoảng cách cosine giữa hai vân giọng, chỉ tính các lượt lớp 3 thật sự chấm. Càng lớn càng chắc; dưới `0,10` là hai giọng quá sát. |
| **Cùng giọng** | `same_voice` — hai bên nghe như một giọng, nhãn phải gán luân phiên theo lượt. |
| **Căn cứ chọn vai** | `role_reason` của lớp 2: vì sao cụm này được coi là agent. Ca bị từ chối thì hiện mã lỗi. |
| **Kết quả** | `ĐẠT` / `KHÔNG ĐẠT` / `TỪ CHỐI ĐÚNG`. |

`TỪ CHỐI ĐÚNG` là **thành công**, không phải lỗi: ca đối chứng là bản ghi hai
kênh tách vai thật, ở đó đoán mò ai với ai là việc thừa và có hại, nên
`separate` phải ném `hai_kenh_that`.

Dòng cuối cho tỉ lệ đạt và **số giây xử lý mỗi phút audio** (kèm RTF = số đó
chia 60). RTF `0,3` nghĩa là một cuộc gọi 10 phút mất khoảng 3 phút máy.

## Ca thử

`cases.json` là danh sách `{file, mo_ta, ky_vong}`:

```json
{
  "file": "/duong/dan/den/cuoc_goi.m4a",
  "mo_ta": "Cuộc gọi thật, hai người nói",
  "ky_vong": { "hai_vai": true, "so_luot_toi_thieu": 4 }
}
```

* `hai_vai: true` — phải tách ra đủ hai vai `caller` và `agent`.
* `so_luot_toi_thieu` — sàn số lượt, để bắt trường hợp gom cả cuộc gọi thành một hai lượt to.
* `loi: "hai_kenh_that"` — ca đối chứng: kỳ vọng `separate` **từ chối** bằng đúng mã đó.

## Lưu ý: tệp audio KHÔNG nằm trong repo

Bản ghi cuộc gọi là dữ liệu thật của người thật, không đẩy lên git. `cases.json`
chỉ giữ **đường dẫn** tới máy đã chạy bộ này. Trên máy khác, các ca đó sẽ báo
`không tìm thấy tệp`.

Cách dùng với mẫu của bạn: bỏ tệp vào một thư mục rồi

```bash
python -m benchmark.run --audio /duong/dan/mau/cua/ban
```

(khớp theo tên tệp), hoặc chép `cases.json` ra chỗ khác, sửa `file` và `ky_vong`
cho đúng bộ của bạn rồi chạy với `--cases`.

Bộ hiện tại gồm mười đoạn cuộc gọi tổng đài thật (~105 giây mỗi đoạn, stereo
giả: hai kênh giống hệt nhau, hai người nói thật) và ba bản dựng hai kênh thật
làm ca đối chứng — chúng phải bị từ chối, vì đã có hai kênh thì không cần tách.
