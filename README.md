# Colab Video POC Skill

把已驗證的 MiniMax H3 Colab 參照圖轉影片 POC，整理成可安裝的 Codex skill 與原創 Python adapter：先檢查環境與工作計畫，再以明確時間及 Compute Units 預算執行一批工作、取回輸出並產生本機 QA。

**驗證狀態：原始 POC 已有單次實測；本 repository 的新 adapter 尚未完成 live GPU 端到端驗證。** 離線測試通過不代表新 adapter 已在 Colab 成功生成。

**English entrypoint:** Original MIT-licensed Codex skill and Python adapter for a MiniMax H3 Colab video POC. Requires Python 3.11+, Linux/macOS, and the official Colab CLI. Start with `python3 scripts/install.py`, then check access with `doctor` and review an offline `plan`. The original POC was tested; this new adapter has not yet been live-GPU tested. Third-party upstream code is fetched at runtime outside this repository and is not covered by MIT. See [THIRD_PARTY.md](THIRD_PARTY.md).

## 功能與範圍

- 每批 1–20 筆工作；每筆使用 1–9 張參照圖片、UTF-8 prompt 檔案，並指定 4–15 秒目標長度。實際輸出依模型的影格對齊規則，未必逐秒精準相等。
- 同一批工作共用這次新建立的 session；批次完成、錯誤或超時後執行停止與清理。
- 後續工作失敗時，保留先前已取回的完成輸出與工作紀錄，讓成功部分可繼續檢查。
- 執行前提供本機計畫；實際 GPU、額度、模型下載及推論仍依 Colab 和上游環境。
- 取回影片後可執行本機媒體驗證；畫面品質與是否符合需求仍須人工觀看。

本次 POC 不包含 2K 驗證。G4 的 96 GB GPU 記憶體是硬體條件，不能據此宣稱它在所有 GPU 中最快。

## 前置條件

1. 本機 Python **3.11 以上**與 Git；adapter、安裝器與 unit tests 使用標準函式庫。
2. Linux 或 macOS；[官方 Colab CLI](https://github.com/googlecolab/google-colab-cli) 目前支援這兩個平台。
3. 已安裝 [uv](https://docs.astral.sh/uv/getting-started/installation/)，並讓 `colab` 位於 `PATH`。
4. 可透過官方 OAuth 流程登入的 Google Colab 帳戶；執行 GPU 工作前確認當次可用硬體、Compute Units 與費用範圍。
5. 本機 `ffmpeg` 與 `ffprobe`，用於影片 QA；參照圖片與 prompt 須有提供給 Colab 處理的權限。
6. `bootstrap` 需要讀取 GitHub；首次遠端準備會下載模型與推論相依套件，須預留時間和儲存空間。

本專案固定官方 CLI 版本，為它使用獨立的 Python 3.12 環境：

```bash
uv tool install --python 3.12 google-colab-cli==0.7.4
colab version
```

Python 3.12 是 CLI 安裝環境；執行本套件的 `python3` 最低為 3.11。套件版本來源：[PyPI 0.7.4](https://pypi.org/project/google-colab-cli/0.7.4/)；`--python` 用法：[uv 官方文件](https://docs.astral.sh/uv/concepts/tools/#python-versions)。

## 安裝 skill

取得本 repository 後，在根目錄執行：

```bash
git clone https://github.com/roy6732856/colab-video-poc-skill.git
cd colab-video-poc-skill
python3 scripts/install.py
```

預設複製到 `~/.codex/skills/colab-video-poc`。安裝只複製本套件的 skill 檔案及授權說明，不登入帳戶、不下載上游、不配置 GPU。

自訂目的地或更新既有安裝：

```bash
python3 scripts/install.py --target ~/.codex/skills
python3 scripts/install.py --target ~/.codex/skills --replace
```

既有 skill 預設保留，安裝器會回傳非零狀態並提示使用 `--replace`。指定替換後，會先準備完整新副本，再把舊版完整保存在目標目錄內的 `.colab-video-poc-backups/`；終端會顯示備份位置。安裝副本排除 `__pycache__`、bytecode、runtime cache、模型、生成媒體及常見憑證檔案。它使用檔案副本，且拒絕會把外部內容帶入套件的 symlink。

以下命令以 repository 根目錄為工作目錄。也可改用已安裝 skill 內的 `scripts/workflow.py` 路徑。完整操作規則見 [SKILL.md](skills/colab-video-poc/SKILL.md)，延伸說明見 [references](skills/colab-video-poc/references/)。

## 一次性 OAuth 登入

在互動式終端執行官方 CLI 的 OAuth 流程：

```bash
colab --auth=oauth2 usage
```

依 CLI 顯示的 Google 授權網址完成登入；若提示輸入授權碼，只在該官方 CLI 的互動提示內輸入。這是帳戶授權與用量查詢，不會建立推論 GPU session。後續依 CLI 的憑證有效狀態重用登入，必要時重新授權。流程來源：[官方認證文件](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md)。

不要把 access token、refresh token、授權碼或帳號資訊貼進 `jobs.json`、prompt、shell 指令、CI、issues 或執行 logs。官方 CLI 自行管理本機憑證；本套件不要求提供 token 字串。公開分享前，也應檢查工作紀錄是否含私人檔案路徑。

## 從檢查到取回影片

先檢查環境與目前帳戶存取。`doctor` 可能透過官方 CLI 查詢帳戶／session 狀態，但不配置 VM，也不自行啟動首次 OAuth 登入：

```bash
python3 skills/colab-video-poc/scripts/workflow.py doctor --json
```

接著準備固定版本的上游依賴。這一步會下載 GitHub 上游到 repository 外的 cache，不配置 GPU：

```bash
python3 skills/colab-video-poc/scripts/workflow.py bootstrap
```

固定 commit 為 `7768ebfb1627546595f37e9acfd2d353937c6cd8`；來源與授權限制見 [THIRD_PARTY.md](THIRD_PARTY.md)。原始上游沒有授權，本套件不重新散布其程式碼、Notebook 或媒體。

依 [manifest 說明](skills/colab-video-poc/references/jobs.md) 建立本機 `jobs.json`，例如：

```json
{
  "jobs": [
    {
      "id": "intro",
      "reference_images": ["inputs/reference.png"],
      "prompt_file": "inputs/prompt.txt",
      "duration_seconds": 8,
      "seed": 20260930
    }
  ]
}
```

路徑相對於 manifest 所在目錄；請自行準備有使用權限的圖片與 UTF-8 prompt。每筆工作需 1–9 張圖片與 4–15 秒目標長度，不要在 manifest 放憑證。先產生本機計畫，確認輸入、輸出位置及工作範圍：

```bash
python3 skills/colab-video-poc/scripts/workflow.py plan --manifest jobs.json --output-dir outputs
```

確認計畫及**本次生成的費用授權**後，再執行 GPU 批次。以下 45 分鐘／8 CU 是明確的範例預算，請依當次核准範圍設定：

```bash
python3 skills/colab-video-poc/scripts/workflow.py run --manifest jobs.json --output-dir outputs --gpu G4 --max-minutes 45 --max-units 8
```

`run` 會建立自己的新 session，在同一 session 依序執行該批工作，取回每筆已完成輸出，並在退出時停止該 session。輸出資料夾必須是新的，或只包含這次審閱的 plan 檔案；已有執行結果時不覆寫。若改變 GPU 或預算，請在新的輸出資料夾重新產生並審閱計畫。帳戶已有 allocation 時會拒絕開始新的批次，需先確認既有工作狀態。

首次環境準備也會消耗時間與額度；Compute Units 的估算、查詢延遲與平台計費不構成固定價格保證。若停止操作因網路或平台錯誤失敗，依工作紀錄確認 session 並透過官方 CLI 或 Colab 管理介面停止，避免持續佔用資源。復原指令見 [setup.md](skills/colab-video-poc/references/setup.md)。

之後對實際取回的影片執行 QA；將 `PATH` 換成影片路徑：

```bash
python3 skills/colab-video-poc/scripts/workflow.py verify --video PATH --output-dir qa
```

媒體探測、解碼與 QA 輸出用於檢查檔案與技術條件；不能替代人工觀看、內容判斷或交付驗收。`outputs/`、`qa/` 與本機工作輸入預設不納入版本控制，repository 不附範例 MP4。

## 已有實測證據

以下為原始 POC 的**一次測試**，不是本 repository 新 adapter 的 live GPU 測試，也不是跨硬體效能比較：

| 項目 | 原始 POC 紀錄 |
| --- | --- |
| 模式 | MiniMax H3 Ref2VA、FP8、4 steps |
| 輸出 | 1376 × 768、8 秒、24 fps |
| ComfyUI 總任務耗時 | 111.82 秒，含編碼、解碼及輸出 |
| 首次準備耗時 | 約 443 秒，另計推論時間 |
| GPU | G4、96 GB GPU 記憶體 |
| 當次觀測用量率 | 8.9 CU／小時 |

用量率為當次觀測值，須以現在帳戶的官方用量資訊為準。這筆結果只支持上述尺寸、模式與長度曾成功執行，不外推為 2K、其他 GPU、所有影片長度或批次工作的成功保證。新 adapter 的驗證範圍目前限於離線檢查；live GPU 驗證仍待執行。完整觀測與 QA 邊界見 [verification.md](skills/colab-video-poc/references/verification.md)。

## 離線測試與貢獻

在根目錄執行 Python 標準函式庫測試：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_*.py' -v
```

[CI](.github/workflows/ci.yml) 在 Linux／macOS、Python 3.11／3.12 執行語法解析與 `unittest`。測試不登入 OAuth、不呼叫 live Colab、不下載模型，也不做付費生成；CI 通過僅代表這些離線檢查通過。本機驗證與安裝讀回紀錄見 [VALIDATION.md](VALIDATION.md)。

提交變更時，保留可重現的離線測試及驗證範圍說明。請勿提交上游 cache、Notebook、模型權重、私人 manifest、帳號、憑證、原始 logs、圖片或生成影片；`.gitignore` 是輔助措施，仍須檢查實際提交內容。

## 授權

本套件原創程式與文件使用 [MIT License](LICENSE)，Copyright © 2026 Ray Li。上游工作流程、模型、獨立安裝工具及使用者素材各自保留原授權，詳見 [THIRD_PARTY.md](THIRD_PARTY.md)。
