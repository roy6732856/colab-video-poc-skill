# 第三方來源與授權邊界

本 repository 的 [MIT License](LICENSE) 僅涵蓋 Ray Li 原創的 skill 指示、adapter、安裝器、測試及文件。第三方程式、Notebook、模型權重、輸入素材與生成媒體不會因本專案採 MIT 而改變授權。

## 執行時取得的上游工作流程

- 來源：[killkli/minimax-h3-colab-skill](https://github.com/killkli/minimax-h3-colab-skill)。
- 固定版本：[`7768ebfb1627546595f37e9acfd2d353937c6cd8`](https://github.com/killkli/minimax-h3-colab-skill/commit/7768ebfb1627546595f37e9acfd2d353937c6cd8)。
- 授權核對紀錄：2026-09-30，GitHub license metadata 為 `null`，且沒有 `LICENSE` 檔案。本紀錄不代表未來版本的授權狀態。
- 公開套件不複製或內嵌該上游的程式碼、Notebook、圖片、影片及其他媒體，也不將其改標為 MIT。
- 原創 adapter 的 `bootstrap` 在使用者明確執行後，以固定 commit 下載執行所需上游到 repository 外的使用者 cache。安裝 skill 與執行離線測試不下載上游。

公開可讀取與可下載不等於獲得重新授權或散布的許可；可參考 GitHub 的 [無授權說明](https://choosealicense.com/no-permission/)。執行時下載也不會授予上游的重新散布、改作或商用權利。若需要把上游一起打包或發布，須先取得適用的明確授權；不得把 cache 加進本套件。

## 獨立安裝的工具與服務

| 元件 | 本專案使用方式 | 授權／條款來源 |
| --- | --- | --- |
| Google 官方 `google-colab-cli==0.7.4` | 使用者透過 `uv tool install --python 3.12` 安裝；adapter 呼叫其 CLI | [官方 repository](https://github.com/googlecolab/google-colab-cli)、[Apache-2.0 License](https://github.com/googlecolab/google-colab-cli/blob/main/LICENSE)、[0.7.4 發行頁](https://pypi.org/project/google-colab-cli/0.7.4/) |
| `uv` | 在本機安裝 CLI 的獨立工具 | [官方授權檔案](https://github.com/astral-sh/uv#license) |
| FFmpeg / ffprobe | 使用者獨立安裝，用於本機媒體解碼、檢查及 QA | [FFmpeg 授權說明](https://ffmpeg.org/legal.html)；實際授權依 build 組態而定 |
| Google Colab | 遠端執行、GPU 與 Compute Units 服務 | [Colab FAQ](https://research.google.com/colaboratory/faq.html) 與帳戶適用的服務條款 |
| GitHub Actions `checkout` / `setup-python` | CI 使用獨立 action 執行離線檢查 | [checkout 授權](https://github.com/actions/checkout/blob/main/LICENSE)、[setup-python 授權](https://github.com/actions/setup-python/blob/main/LICENSE) |

本套件未內嵌這些工具的程式或執行環境；各工具及其相依套件保留自己的授權。第三方推論相依套件由遠端工作流程取得，須依實際安裝版本核對其授權與 notices。

## 模型、素材與輸出

MiniMax H3 與工作流程使用的其他模型、量化版本及模型相依資產，均保留各自的 model card、`LICENSE` 與使用限制。本套件不散布模型權重，不把模型標為 MIT，也不保證特定模型用途已取得授權。下載與使用前，應閱讀實際選用版本的來源及條款；模型入口為 [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)，工作流程可參考 [ComfyUI 官方 H3 說明](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)。

參照圖片、prompt 與生成影片的權利及公開使用範圍另行確認。公開 repository 不提供原始 POC 的圖片或 MP4，不保存帳號資訊、OAuth 憑證、session tokens、私人工作 manifest 或可識別的執行 logs。自行分享 QA 或除錯紀錄前，也應移除私人路徑、帳號與憑證。
