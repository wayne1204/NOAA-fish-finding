import xml.dom
import xml.dom.minidom


def createElementNode(doc, tag, attr): 
    element_node = doc.createElement(tag)
    text_node = doc.createTextNode(attr)
    element_node.appendChild(text_node)
    return element_node


def createChildNode(doc, tag, attr, parent_node):
    child_node = createElementNode(doc, tag, attr)
    parent_node.appendChild(child_node)


def createObjectNode(doc, attr, width, height):
    object_node = doc.createElement('object')
    createChildNode(doc, 'name', attr[1],
                    object_node)
    createChildNode(doc, 'pose',
                    'Unspecified', object_node)
    createChildNode(doc, 'truncated',
                    '0', object_node)
    createChildNode(doc, 'difficult',
                    '0', object_node)

    bndbox_node = doc.createElement('bndbox')
    # if attr[2] < 0 or attr[3] < 0:
    #     print('xmin: {:.4f} | ymin: {:.4f}'.format(attr[2], attr[3]))
    # if attr[2] + attr[4] > width: 
    #     print('xmin {:.4f} xmax: {:.4f} / {}'.format(attr[2], attr[2]+attr[4], width))
    # if attr[3] + attr[5] > height:
    #     print('ymin {:.4f} ymax: {:.4f} / {}'.format(attr[3], attr[3]+attr[5], height))
    xmin = str(max(attr[2], 0))
    ymin = str(max(attr[3], 0))
    xmax = str(min(attr[2]+attr[4], width))
    ymax = str(min(attr[3]+attr[5], height))

    createChildNode(doc, 'xmin', xmin, bndbox_node)
    createChildNode(doc, 'ymin', ymin, bndbox_node)
    createChildNode(doc, 'xmax', xmax, bndbox_node)
    createChildNode(doc, 'ymax', ymax, bndbox_node)
    object_node.appendChild(bndbox_node)
    return object_node


def createRootNode(dataset, fname, width, height, channel, ):
    my_dom = xml.dom.getDOMImplementation()
    doc = my_dom.createDocument(None, 'annotation', None)

    root_node = doc.documentElement
    createChildNode(doc, 'folder', dataset, root_node)
    createChildNode(doc, 'filename', fname, root_node)

    # source
    source_node = doc.createElement('source')
    createChildNode(doc, 'database', 'NOAA_fish_finding', source_node)
    createChildNode(doc, 'annotation', '_ANNOTATION', source_node)
    createChildNode(doc, 'image', 'flickr',  source_node)
    createChildNode(doc, 'flickrid', 'NULL', source_node)
    root_node.appendChild(source_node)

    # owner
    owner_node = doc.createElement('owner')
    createChildNode(doc, 'flickrid', 'NULL', owner_node)
    createChildNode(doc, 'name', "_AUTHOR",  owner_node)
    root_node.appendChild(owner_node)

    # size
    size_node = doc.createElement('size')
    createChildNode(doc, 'width',  str(width),   size_node)
    createChildNode(doc, 'height', str(height),  size_node)
    createChildNode(doc, 'depth',  str(channel), size_node)
    root_node.appendChild(size_node)

    # segmented
    createChildNode(doc, 'segmented', '0', root_node)
    return doc, root_node


def writeXMLFile(doc, filename):
    tmpfile = open('tmp.xml', 'w')
    doc.writexml(tmpfile, addindent=''*4, newl='\n', encoding='utf-8')
    tmpfile.close()
    fin = open('tmp.xml')
    fout = open(filename, 'w')

    lines = fin.readlines()

    for line in lines[1:]:
        if line.split():
            fout.writelines(line)

    fin.close()
    fout.close()