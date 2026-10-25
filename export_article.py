"""Export the article source to GitHub-readable Markdown using only the standard library."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    content = json.loads((ROOT / 'article_content.json').read_text())
    lines = [f"# {content['title']}", content['subtitle'],
             f"**{content['author']}** · {content['handle']}"]
    pages = content['pages'] + [content['appendix']]
    for index, page in enumerate(pages):
        if index == len(content['pages']):
            lines.append('<details>\n<summary>Optional technical note and experiment details</summary>')
        for item in page:
            kind = item['type']
            if kind == 'p':
                lines.append(item['text'])
            elif kind == 'h':
                lines.append('## ' + item['text'])
            elif kind == 'figure':
                image = ROOT / 'figures' / (item['name'] + '.png')
                assert image.is_file(), image
                lines.append(f"![{item['caption']}](../figures/{image.name})\n\n{item['caption']}")
            elif kind == 'sources':
                lines.append('### ' + item['text'])
                lines.append('\n'.join(f"- [{s['label']}]({s['url']})" for s in content['sources']))
                lines.append(content['reproducibility'])
    lines.append('</details>')
    target = ROOT / 'docs/article.md'
    target.parent.mkdir(exist_ok=True)
    target.write_text('\n\n'.join(lines) + '\n')
    print(target)


if __name__ == '__main__':
    main()
