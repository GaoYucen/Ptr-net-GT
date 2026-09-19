# GroupOpt ICLR 2027 论文目录

本目录基于 ICLR 官方发布的 2027 LaTeX 模板建立。官方样式文件保持原样，
论文内容从 `main.tex` 和 `sections/` 中继续编写。

## 目录说明

- `main.tex`：论文入口、匿名设置、章节顺序和声明。
- `sections/`：正文与附录的分章节源文件。
- `references.bib`：经过核验的参考文献条目。
- `iclr2027_conference.sty`：ICLR 2027 官方样式，不要修改。
- `iclr2027_conference.bst`：ICLR 2027 官方参考文献样式，不要修改。
- `iclr2027_conference.tex`：官方完整格式说明和示例，保留作参考。
- `math_commands.tex`、`fancyhdr.sty`、`natbib.sty`：官方模板依赖。

## 编译

在本目录执行：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

清理构建文件：

```bash
latexmk -C -outdir=build main.tex
```

## 投稿前检查

- 投稿版本保持 `\iclrfinalcopy` 注释状态。
- 正文不超过 9 页；参考文献和附录不计入正文页数。
- 作者、单位、致谢、代码链接和文件元数据均不得泄露身份。
- 摘要必须为单段。
- 必须完成 AI use statement。
- 建议提供 Reproducibility statement。
- 不修改官方样式文件、字号、页边距或版面参数。

官方说明：<https://iclr.cc/Conferences/2027/AuthorGuidelines>

