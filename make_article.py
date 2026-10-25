"""Build the editable article and a self-contained reading copy."""
import base64
import html
import json
from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

ROOT=Path(__file__).resolve().parent
C=json.loads((ROOT/'article_content.json').read_text())


def hyperlink(paragraph, label, url):
    rid=paragraph.part.relate_to(url,RT.HYPERLINK,is_external=True)
    link=OxmlElement('w:hyperlink');link.set(qn('r:id'),rid)
    run=OxmlElement('w:r');prop=OxmlElement('w:rPr')
    color=OxmlElement('w:color');color.set(qn('w:val'),'176D70');prop.append(color)
    run.append(prop);text=OxmlElement('w:t');text.text=label;run.append(text)
    link.append(run);paragraph._p.append(link)


def alt_text(run, text):
    for dp in run._r.xpath('.//wp:docPr'):
        dp.set('descr',text)


def docx():
    doc=Document()
    doc.core_properties.title=C['title']
    doc.core_properties.author=C['author']
    doc.core_properties.subject='Learning during use, adaptive model weights and the learning–forgetting tradeoff'
    sec=doc.sections[0]
    sec.page_width=Inches(8.5);sec.page_height=Inches(11)
    sec.top_margin=Inches(.65);sec.bottom_margin=Inches(.65)
    sec.left_margin=Inches(.78);sec.right_margin=Inches(.78)
    sec.header_distance=Inches(.25);sec.footer_distance=Inches(.28)
    normal=doc.styles['Normal'];normal.font.name='Calibri';normal.font.size=Pt(11)
    normal.font.color.rgb=RGBColor.from_string('182C3A')
    normal.paragraph_format.space_after=Pt(8)
    normal.paragraph_format.line_spacing=1.12
    normal.paragraph_format.widow_control=True
    for key,size in [('Title',27),('Subtitle',12),('Heading 1',17),('Heading 2',12),('Header',8)]:
        st=doc.styles[key];st.font.name='Calibri';st.font.size=Pt(size)
        st.font.color.rgb=RGBColor(0,0,0)
        st.font.underline=False
        st.paragraph_format.space_before=Pt(0 if key in ['Title','Subtitle'] else 9)
        st.paragraph_format.space_after=Pt(7)
        if key.startswith('Heading'): st.paragraph_format.keep_with_next=True
        if st._element.pPr is not None:
            for b in list(st._element.pPr.findall(qn('w:pBdr'))):st._element.pPr.remove(b)
    doc.styles['Caption'].font.name='Calibri';doc.styles['Caption'].font.size=Pt(9)
    doc.styles['Caption'].font.bold=False
    doc.styles['Caption'].font.italic=False
    doc.styles['Caption'].font.color.rgb=RGBColor.from_string('526570')
    doc.styles['Caption'].paragraph_format.space_after=Pt(9)
    doc.styles['Caption'].paragraph_format.line_spacing=1.05
    header=sec.header.paragraphs[0]
    header.text='LEARNING DURING USE'
    header.style='Header'
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    footer.paragraph_format.space_after=Pt(0)
    run=footer.add_run(C['author']+'   •   ');run.font.size=Pt(8);run.font.color.rgb=RGBColor.from_string('526570')
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    doc.add_paragraph(C['title'].replace(' While ', '\nWhile '),'Title')
    doc.add_paragraph(C['subtitle'],'Subtitle')
    by=doc.add_paragraph(C['author']+'  |  '+C['handle'])
    by.paragraph_format.space_after=Pt(12)
    for run in by.runs:run.font.size=Pt(9);run.font.color.rgb=RGBColor.from_string('526570')
    for i,page in enumerate(C['pages']+[C['appendix']]):
        first_paragraph_index=len(doc.paragraphs)
        for item in page:
            if item['type']=='p':doc.add_paragraph(item['text'])
            elif item['type']=='h':doc.add_paragraph(item['text'],'Heading 1')
            elif item['type']=='figure':
                p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before=Pt(2);p.paragraph_format.space_after=Pt(3)
                p.paragraph_format.keep_with_next=True
                run=p.add_run();run.add_picture(str(ROOT/'figures'/(item['name']+'.png')),width=Inches(6.65))
                alt_text(run,item['caption'])
                doc.add_paragraph(item['caption'],'Caption')
            elif item['type']=='sources':
                doc.add_paragraph(item['text'],'Heading 2')
                for source in C['sources']:
                    p=doc.add_paragraph();p.paragraph_format.space_after=Pt(3)
                    p.style='Caption';hyperlink(p,source['label'],source['url'])
                p=doc.add_paragraph(C['reproducibility'],'Caption')
                p.paragraph_format.space_before=Pt(4)
        if i:
            doc.paragraphs[first_paragraph_index].paragraph_format.page_break_before=True
    path=ROOT/'When-an-LLM-learns-during-use.docx'
    doc.save(path)
    return path


def html_article():
    esc=html.escape
    css='''
    :root{--ink:#142b3c;--muted:#526570;--paper:#faf8f2;--teal:#087f7a}
    *{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink)}
    article{max-width:870px;padding:64px 32px 84px;margin:auto}header{margin-bottom:28px}
    .eyebrow{font:700 12px/1.5 system-ui;letter-spacing:.12em;color:var(--teal)}
    h1{font:700 clamp(36px,6vw,56px)/1.08 system-ui;letter-spacing:-.035em;margin:18px 0;text-wrap:balance}
    .subtitle{font:400 22px/1.45 system-ui;color:var(--muted);margin:0 0 20px}
    .byline{font:500 14px/1.5 system-ui;color:var(--muted)}
    p{font:400 20px/1.72 Georgia,serif;margin:0 0 22px}
    h2{font:700 28px/1.25 system-ui;letter-spacing:-.025em;margin:48px 0 20px}
    figure{margin:30px -18px 26px}img{display:block;width:100%;height:auto}
    figcaption{font:400 13px/1.55 system-ui;color:var(--muted);margin:12px 18px}
    a{color:#006c71;text-decoration-thickness:1px;text-underline-offset:3px}
    .sources{border-top:1px solid #d9dfda;margin-top:40px;padding-top:18px}
    .sources p,.sources li{font:400 14px/1.65 system-ui;color:var(--muted)}
    .sources h2{font-size:20px;margin:12px 0}li{margin-bottom:7px}
    details{border-top:1px solid #d9dfda;margin-top:38px;padding-top:22px}
    summary{cursor:pointer;font:600 18px/1.5 system-ui;color:var(--teal)}
    details[open] summary{margin-bottom:25px}
    details h2{font-size:23px}details p{font-size:17px}
    @media(max-width:540px){article{padding:32px 22px}p{font-size:18px}figure{margin-left:-14px;margin-right:-14px}}
    @media print{article{max-width:100%;padding:0}p{font-size:11pt}h1{font-size:28pt}figure{break-inside:avoid}h2{break-after:avoid}}
    '''
    parts=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
           '<title>'+esc(C['title'])+'</title><meta name="description" content="'+esc(C['description'])+'">',
           '<style>'+css+'</style></head><body><article><header><div class="eyebrow">LEARNING DURING USE / AN IDEA PUT TO THE TEST</div>',
           '<h1>'+esc(C['title'])+'</h1><div class="subtitle">'+esc(C['subtitle'])+'</div>',
           '<div class="byline">'+esc(C['author'])+' · '+esc(C['handle'])+'</div></header>']
    for page_index,page in enumerate(C['pages']+[C['appendix']]):
        if page_index==len(C['pages']):
            parts.append('<details><summary>Optional technical note and experiment details</summary>')
        for item in page:
            typ=item['type']
            if typ=='p':parts.append('<p>'+esc(item['text'])+'</p>')
            elif typ=='h':parts.append('<h2>'+esc(item['text'])+'</h2>')
            elif typ=='figure':
                b64=base64.b64encode((ROOT/'figures'/(item['name']+'.png')).read_bytes()).decode()
                parts.append('<figure><img src="data:image/png;base64,'+b64+'" alt="'+esc(item['caption'])+'"><figcaption>'+esc(item['caption'])+'</figcaption></figure>')
            elif typ=='sources':
                parts.append('<section class="sources"><h2>'+esc(item['text'])+'</h2><ul>')
                for source in C['sources']:parts.append('<li><a href="'+esc(source['url'])+'">'+esc(source['label'])+'</a></li>')
                parts.append('</ul><p>'+esc(C['reproducibility'])+'</p></section>')
    parts.append('</details></article></body></html>')
    path=ROOT/'When-an-LLM-learns-during-use.html'
    path.write_text(''.join(parts))
    return path


if __name__=='__main__':
    print(docx());print(html_article())
    words=sum(len(x.get('text','').split()) for page in C['pages'] for x in page if x['type'] in ['p','h'])
    print(f'Main article body and headings: {words} words; technical appendix is separate')
