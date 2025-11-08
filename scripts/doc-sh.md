> `scripts/run.sh` 是一个自动化运行脚本，用来在干净环境下完成：创建虚拟环境、安装依赖、下载数据、训练 baseline 模型、训练若干消融实验（自动化命名输出目录）、保存训练产物（模型、vocab、train_loss.png）并将 Markdown 报告转换为 PDF（需要系统上安装 pandoc + LaTeX）。它让你能用一条命令复现所有关键实验和生成可提交的 artifacts（结果目录与 PDF 报告）。

> 注意：用户环境是 Windows 11。`run.sh` 是 shell 脚本，适用于 Linux/macOS/WSL。若你在 Windows 原生 CMD/PowerShell，请参照下面的 Windows 使用说明或使用 WSL（推荐）。
>
> ---
>
> ## 在 Windows 上如何运行？
>
> 你有三种选择（按优先级）：
>
> 1. **使用 WSL（推荐）**
>    * 在 Windows 上启用 WSL（例如 Ubuntu），打开 WSL 终端，进入项目目录，然后运行：
>    * ```
>      bash scripts/run.sh
>      ```
>    * 效果与 Linux 相同。
> 2. **使用 PowerShell / CMD（直接在 Windows 原生执行）**
>    * 我也给出等价命令片段，你可以把这些命令保存为 `scripts/run.bat` 或在 PowerShell 中逐条执行。关键点：
>
>      * 使用 `python -m venv venv_run` 创建虚拟环境。
>      * 激活 `venv_run\Scripts\activate`（PowerShell: `venv_run\Scripts\Activate.ps1`）。
>      * `pip install` 依赖。
>      * 依次运行 `python train.py ...`（见上面每个实验的 python 命令）。
>    * Windows 示例（PowerShell）：
>
>      ```
>      python -m venv venv_run
>      .\venv_run\Scripts\Activate.ps1
>      pip install --upgrade pip
>      pip install torch matplotlib requests tqdm
>      python - <<'PY'
>      from data import download_tiny_shakespeare
>      download_tiny_shakespeare('data/tiny_shakespeare.txt')
>      PY
>      # Run baseline
>      python train.py --task seq2seq --data data/tiny_shakespeare.txt --seq_len 128 --batch_size 32 --epochs 6 --use_pos_encoding --relative_pos --save results\seq2seq_base --seed 42
>      # and so on for ablation commands...
>
>      ```
> 3. **手动逐条运行（最安全）**
>    * 如果你不想用脚本，可以手动执行脚本里每一步（创建 venv、安装依赖、运行 train.py 的命令）。这是最不容易出错但最慢的方法。
