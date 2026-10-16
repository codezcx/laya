"""SEED-VII label reader; historical model and evaluation removed."""
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
SOURCE=Path('/inspire/hdd/global_user/zhangchenxi-253108310322/mne_data/SEED-VII')

def labels():
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(SOURCE/'emotion_label_and_stimuli_order.xlsx') as z:
  strings=[''.join(n.itertext()) for n in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)]
  table=[]
  for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//s:row',ns):
   vals=[]
   for c in row:
    v=c.find('s:v',ns)
    if v is not None:vals.append(strings[int(v.text)] if c.attrib.get('t')=='s' else v.text)
   table.append(vals)
 mapping={}
 for i in range(0,8,2):
  for index,label in zip(table[i][1:],table[i+1][1:]):mapping[int(index)]=label
 assert len(mapping)==80
 return mapping
