#coding=utf-8
import  xml.dom.minidom

#打开xml文档
dom = xml.dom.minidom.parse('./date_demo/ans/000001.xml')
root = dom.documentElement

print(root.getElementsByTagName('width')[0].firstChild.data)
print(root.getElementsByTagName('height')[0].firstChild.data)
print(root.getElementsByTagName('name')[0].firstChild.data)
print(root.getElementsByTagName('xmin')[0].firstChild.data)
print(root.getElementsByTagName('ymin')[0].firstChild.data)
print(root.getElementsByTagName('xmax')[0].firstChild.data)
print(root.getElementsByTagName('ymax')[0].firstChild.data)
