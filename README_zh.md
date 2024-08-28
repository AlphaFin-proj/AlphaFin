# AlphaFin: 基于检索增强股票链框架的金融分析基准

<p align="center">
    <img src="assets/readme_logo.png" width="40%"/>
</p>

## 免责声明

本仓库内容仅供**学术研究和教育用途**。尽管StockGPT提供各种任务和场景下的金融服务，但模型生成的内容仅供用户参考，不应被视为金融、法律或投资建议。作者和贡献者不对StockGPT生成信息的准确性、完整性或实用性负责，用户在做出任何金融、法律或投资决策前，应自行判断并寻求专业建议。使用本仓库的软件和信息由用户自行承担风险。

**使用或访问本仓库中的信息即表示您同意赔偿、辩护并使作者、贡献者及任何相关组织或个人免于任何及所有索赔或损害。**

## 简介

<p align="center">
    <img src="assets/case.png" width="50%"/>
</p>

我们开源了**AlphaFin**系列，包括**AlphaFin数据集**，在AlphaFin上训练的聊天模型，即**StockGPT-Stage1**和**StockGPT-Stage2**，以及**Stock-Chain**，这是一个检索增强的金融分析框架。

我们专注于两个实际金融任务：**股票趋势预测**和**金融问答**。通过集成RAG，我们解决了大语言模型输出的幻觉问题及其生成实时内容的能力不足的问题。

- [AlphaFin](https://huggingface.co/datasets/AlphaFin/AlphaFin-dataset-v1)：包含传统研究数据集、实时金融数据和手写的CoT数据，增强了大语言模型在金融分析中的能力。

- [StockGPT-Stage1](https://huggingface.co/AlphaFin/StockGPT-Stage1)：使用LoRA方法在AlphaFin的金融报告和金融报告CoT上微调的大语言模型，专用于股票趋势预测任务。

- [StockGPT-Stage2](https://huggingface.co/AlphaFin/StockGPT-Stage2)：继续使用研究数据集、金融新闻和AlphaFin的StockQA数据集训练，更加全面，能够处理金融问答任务。

## 快速开始

准备

```bash
git clone https://github.com/AlphaFin-proj/AlphaFin.git
cd AlphaFin
pip install -r requirements.txt
```

阶段1

```bash
bash scripts/stage1_trend_prediction.sh
```

阶段2

```bash
bash scripts/stage2_financial_qa.sh
```

对于阶段2，我们提供200个新闻、研究报告和股票价格文档的样本数据供您尝试该项目，更多的文档数据将在整理后尽快开放。

## 数据集

<p align="center">
    <img src="assets/datasource.png" width="600"/>
</p>

如图所示，展示了AlphaFin数据集的数据来源和预处理。我们确保数据集涵盖广泛的核心金融分析任务，包括NLI、金融问答、股票趋势预测等。AlphaFin包含中英文数据集，以消除潜在的语言偏见。英文数据主要包括传统的金融和NLP相关任务，而中文数据则主要包含金融研究报告和股票预测。

| 数据集 | 大小 | 输入长度  | 输出长度 | 语言 |
| - | - | - | - | - |
| Research | 42K | 712.8 | 5.6 | 英文 |
| StockQA | 21K | 1313.6 | 40.8 | 中文 |
| Fin. News | 79K | 497.8 | 64.2 | 中文 |
| Fin. Reports | 120K | 2203 | 17.2 | 中文 |
| Fin. Reports CoT | 200 | 2184.8 | 407.8 | 中文 |

## 性能

<p align="center">
    <img src="assets/long-short-test.png" width="1000"/>
</p>

我们的Stock-Chain在2023年起实现了最高的年化回报率，并保持上升趋势。这表明Stock-Chain在投资中的有效性。Stock-Chain实现了30.8%的最高年化回报率，证明了其有效性。

## 金融问答案例

![image](assets/case_travel.png)

## 引用

如果您在您的工作中使用了AlphaFin，请引用我们的论文。

```
@misc{li2024alphafin,
      title={AlphaFin: Benchmarking Financial Analysis with Retrieval-Augmented Stock-Chain Framework}, 
      author={Xiang Li and Zhenyu Li and Chen Shi and Yong Xu and Qing Du and Mingkui Tan and Jun Huang and Wei Lin},
      year={2024},
      eprint={2403.12582},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
