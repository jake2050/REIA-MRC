#!/usr/bin/env python3
"""
@author: 孙伟伟
@contact: jakesun2020@163.com
@application:
@file: 预训练.py
@time: 2021/10/24/024 16:44
"""
import os
import re
import argparse

from tqdm import tqdm
from constants import *


def aligment_ann(original, newtext, ann_file, offset):
    '''
    raw_txt[offset:], ntxt2, ann_path, offset
    aligment_ann(ann文件的原始文本, 清理过的ann文件, 经过for遍历得到的ann文件名, 被清理的文本内容在原始数据中的偏移)
    Args:
        original: raw_txt[offset:] 原始文本，而不是经过分词然后转化为字符串的文本。
            like many heartland states , iowa has had trouble keeping young people down on the farm or anywhere within state lines .
        newtext: 原始文本经过BERT分词又经过to_string的文本。               两个文本不一样,区别在于bert分词后，逗号也被分为一个词，因此多了很多空格，因为由分词合成字符串时，每个分词都会出现空格
            like many heartland states , iowa has had trouble keeping young people down on the farm or anywhere within state lines .
        ann_file: ann文件的文件名
            '/home/wgy/paper/REIA/data/raw_data/ACE2004/train0/ABC20001001.1830.0973.ann'
        offset: 偏移，65

    Returns: entities1, relations1
     # entities1 = ['LOC',463, 467, 'area']
    # relations1 = [PART-WHOLE, rel_el_idx, rel_e2_idx]
    '''
    # Ensure that uncased tokenizers can also be aligned
    original = original.lower()
    newtext = newtext.lower()
    annotation = []# 不是以T开头的数据集即实体数据集，保存的是关系
    # annotation =
    #   ['R1-1\tPHYS Arg1:T22-1 Arg2:T43-82\n',
    #   'R2-1\tPHYS Arg1:T44-93 Arg2:T48-94\n',
    #   'R2-2\tPHYS Arg1:T44-90 Arg2:T48-91\n',
    #   'R4-1\tEMP-ORG Arg1:T51-97 Arg2:T50-96\n',
    #   ...]
    terms = {}         # 保存的是实体在正文中的相对位置
    # {913: [[913, 920, 'T1-27', 'GPE', 'america']],
    #  944: [[944, 951, 'T1-28', 'GPE', 'america']],
    #  328: [[328, 331, 'T1-101', 'GPE', 'usa']],
    #  365: [[365, 370, 'T2-3', 'PER', '2,000']],
    #  357: [[357, 361, 'T3-2', 'GPE', 'town']], ... ,
    #  }

    ends = {}          #ends:{ 920: [913], 951: [944], 331: [328], 370: [365], 361: [357], 379: [377], 414: [402],...,}
    with open(ann_file) as file:# 'T1-27	GPE 978 985	america'
        for line in file.readlines():
            if line.startswith('T'):
                annots = line.rstrip().split("\t", 2)# 以\t 分割为三段
                # annots: ['T1-27', 'GPE 978 985', 'america']
                typeregion = annots[1].split(" ")
                # typeregion: ['GPE', '978', '985']
                start = int(typeregion[1]) - offset # start: 913                实体在正文中的相对位置,原始文本
                end = int(typeregion[2]) - offset # end: 920
                if not start in terms:
                    terms[start] = [] # terms: {913:[]}
                if not end in ends:
                    ends[end] = [] # ends: {920:[]}
                if len(annots) == 3:
                    terms[start].append(
                        [start, end, annots[0], typeregion[0], annots[2]])\
                    # {913: [[913, 920, 'T1-27', 'GPE', 'america']]}
                else:
                    terms[start].append([start, end, annots[0], typeregion[0], ""])
                    # 在实体对齐的时候，以T开头的实体没有分为三段(T, (start,end), str),把最后一位设为空字符串
                ends[end].append(start)
                # {920: [913]}
            else:# 不是以“T”作为开始的文件名,即关系类型和头实体 尾实体 ground truth
                annotation.append(line)
    orgidx = 0
    newidx = 0
    orglen = len(original)# 2221 去掉文件头的正文文本长度
    newlen = len(newtext)# 2262  分词再组合的文本长度
    # 给每个实体分配newtxt的相对位置          {913: [[926, 933, 'T1-27', 'GPE', 'america']], 944: [[957, 964, 'T1-28', 'GPE', 'america']], 328: [[330, 333, 'T1-101', 'GPE', 'usa']],...}
    while orgidx < orglen and newidx < newlen:# original 和 newtxt都是字符串
        if original[orgidx] == newtext[newidx]:# original[0] = "l" original[1] = "i" original[2] = "k"
            orgidx += 1
            newidx += 1
        elif newtext[newidx] == '\n':
            newidx += 1
        elif original[orgidx] == '\n':
            orgidx += 1
        elif newtext[newidx] == ' ':
            newidx += 1
        elif original[orgidx] == ' ':
            orgidx += 1
        elif newtext[newidx] == '\t':
            newidx += 1
        elif original[orgidx] == '\t':
            orgidx += 1
        elif newtext[newidx] == '.':
            # ignore extra "." for stanford
            newidx += 1
        else:# 断言是处理异常的
            assert False, "%d\t$%s$\t$%s$" % (# 当全部不满足时，执行断言后面的内容，当断言条件为False（异常），执行断言后面的程序
                orgidx, original[orgidx:orgidx + 20], newtext[newidx:newidx + 20])
        if orgidx in terms:
            for l in terms[orgidx]:# terms[10] = [[10, 19, 'T43-82', 'LOC', 'heartland']] l = [10, 19, 'T43-82', 'LOC', 'heartland']
                # {913: [[913, 920, 'T1-27', 'GPE', 'america']],
                #  944: [[944, 951, 'T1-28', 'GPE', 'america']],
                #  328: [[328, 331, 'T1-101', 'GPE', 'usa']],
                #  365: [[365, 370, 'T2-3', 'PER', '2,000']],
                #  357: [[357, 361, 'T3-2', 'GPE', 'town']], ... ,
                #  }
                l[0] = newidx
                # l[0] = [10, 19, 'T43-82', 'LOC', 'heartland']
        if orgidx in ends:
            for start in ends[orgidx]:
                for l in terms[start]:
                    if l[1] == orgidx:
                        l[1] = newidx
            del ends[orgidx]
    entities = []
    relations = []
    dict1 = {}
    # '{    T1-27': 0,
        # 'T1-28': 1,
        # 'T1-101': 2,
        # 'T2-3': 3,
        # 'T3-2': 4,
        # 'T3-4': 5,
    # }
    i = 0
    for ts in terms.values():
        # ts: [[926, 933, 'T1-27', 'GPE', 'america']]
        for term in ts:
            # term:[926, 933, 'T1-27', 'GPE', 'america']
            if term[4] == "":
                entities.append([term[2], term[3], term[0],
                                 term[1], newtext[term[0]:term[1]]])
            else:
               # &AMP在HTML中表示&符号
                assert newtext[term[0]:term[1]].replace(" ", "").replace('\n', "").replace("&AMP;", "&").replace("&amp;", "&") == \
                    term[4].replace(" ", "").lower(
                ), newtext[term[0]:term[1]] + "<=>" + term[4]
                # newtext[term[0]:term[1]] newtext[start:end]实体在newtext中的tokenid
                entities.append([term[2], term[3], term[0], term[1],
                                 newtext[term[0]:term[1]].replace("\n", " ")])# entities = [['T1-27', 'GPE', 926, 933, 'america'], ['T1-28', 'GPE', 957, 964, 'america'], ['T1-101', 'GPE', 330, 333, 'usa'], ['T2-3', 'PER', 367, 374, '2 , 000'], ['T3-2', 'GPE', 359, 363, 'town']]
            dict1[term[2]] = i
            # dict = {'T1-27': 0, 'T1-28': 1, 'T1-101': 2, 'T2-3': 3, 'T3-2': 4}# dict表示把ground truth的实体按行编号，第一个实体T1-27编号为0，后期通过ground truth的关系中的Arg1索引到第T那一行的实体
            i += 1
    for rel in annotation:
        # annotation =
        #   ['R1-1\tPHYS Arg1:T22-1 Arg2:T43-82\n',
        #   'R2-1\tPHYS Arg1:T44-93 Arg2:T48-94\n',
        #   'R2-2\tPHYS Arg1:T44-90 Arg2:T48-91\n',
        #   'R4-1\tEMP-ORG Arg1:T51-97 Arg2:T50-96\n',
        #   ...]
        rel_id, rel_type, rel_e1, rel_e2 = rel.strip().split()
        # rel_e1 = 'Arg1:T22-1',
        # rel_e2 = 'Arg2:T43-82'
        # rel_id = 'R1-1'
        # rel_type = 'PHYS'
        rel_e1 = rel_e1[5:]
        rel_e2 = rel_e2[5:]
        # rel_e1 = 'T22-1',
        # rel_e2 = 'T43-82'
        relations.append([rel_id, rel_type, rel_e1, rel_e2])
        # relations = [
        #   ['R1-1', 'PHYS', 'T22-1', 'T43-82'],
        #   ['R2-1', 'PHYS', 'T44-93', 'T48-94'],
        #   ['R2-2', 'PHYS', 'T44-90', 'T48-91'],
        #   ['R4-1', 'EMP-ORG', 'T51-97', 'T50-96'],
        #   ['R4-2', 'EMP-ORG', 'T51-81', 'T50-83'], ...,
        #   ]
    relations1 = []
    for rel in relations:
        _, rel_type, rel_e1, rel_e2 = rel
        rel_e1_idx = dict1[rel_e1]# 通过T?-?索引到第几个实体，这个实体是头实体
        rel_e2_idx = dict1[rel_e2]
        relations1.append([rel_type, rel_e1_idx, rel_e2_idx])# (关系类型, 在ground truth中头实体是第几个实体, 尾实体是第几个实体)
        # relations1 = [
        #               ['PHYS', 44, 84], ['PHYS', 90, 94], 
        #               ['PHYS', 88, 93], ['EMP-ORG', 99, 97],...,
        #          ]
    entities1 = [[ent[1], ent[2], ent[3], ent[4]] for ent in entities]# entities = [['T1-27', 'GPE', 926, 933, 'america']]
    return entities1, relations1


def passage_blocks(txt, window_size, overlap):
    # newtxt: 15 dead as suicide bomber blasts student bus, Israel hits back in Gaza
    # HAIFA , Israel , March 5 ( AFP ) - Fifteen people were killed and more than 30 wounded Wednesday as a suicide bomber blew himsel

    # txt: ['15', 'dead', 'as', 'suicide', 'bomber', ...]
    blocks = []
    regions = []
    for i in range(0, len(txt), window_size-overlap):# 相当于池化操作，
        b = txt[i:i+window_size]# 把清洗过的数据以windows_size的长度进行分块
        blocks.append(b)# blocks存储的是以windows_size长度数据的列表
        regions.append((i, i+window_size))# 数据存储的区域
    return blocks, regions


def get_block_er(txt, entities, relations, window_size, overlap, tokenizer):
    """
    Get the block level annotation获取块级注释
    Args:
        txt: text to be processed, list of token被清理过的数据
        entities: list of (entity_type, start, end,entity)转化为piece的实体
        relations: list of (relation_type,entity1_idx,entity2_idx)
        window_siez: sliding window size
        overlap: overlap between two adjacent windows
    Returns:
        ber: list of [block，entities, relations]
    """
    blocks, block_range = passage_blocks(txt, window_size, overlap)
    # blocks： 把一个训练样本分为几块，大小为window_size，块与块之间的重合大小为overlap，          第一个txt文本分为了两块
        # 存储的为block_range范围内的字符
    # block_range：[(0, 300), (285, 585)]
    ber = [[[], [], []] for i in range(len(block_range))]
    #
    #                       ['like', 'many', 'heartland', 'states',], [('GPE', 198, 199, 'america'), ('GPE', 205, 206, 'america'),],[]
    # ber = [      [[],[],[]],
    #              [[],[],[]],
    #       ]


    e_dict = {}## e_dict第j个实体在第i个块中{0: [0], 1: [0], 2: [0], 3: [0]}

    for i, (s, e) in enumerate(block_range):
        es = []#[('GPE', 198, 199, 'america'), ('GPE', 205, 206, 'america'), ('GPE', 65, 66, 'usa'), ('PER', 74, 77, '2 , 000')]
        for j, (entity_type, start, end, entity_str) in enumerate(entities):# entities表示一个训练文本中标注的所有实体，并且实体的start和end的id是以word进行划分的，第一个单词的标号为0，依次标号
            if start >= s and end <= e:
                nstart, nend = start-s, end-s # nstart:198 nend:199  # nstart和nend都是相对于每一块中的第一个单词的位置
                if tokenizer.convert_tokens_to_string(blocks[i][nstart:nend]) == entity_str:
                    es.append((entity_type, nstart, nend, entity_str))
                    e_dict[j] = e_dict.get(j, [])+[i]# dict.get("apple", "没有apple"): 如果字典有“apple"这个key,那么返回对应的value，如果没有，则返回"没有apple"
                    # e_dict第j个实体在第i个块中{0: [0], 1: [0], 2: [0], 3: [0]}
                else:
                    print(
                        "The entity string and its corresponding index are inconsistent\n")
        # blocks： 把一个训练样本分为几块，大小为window_size，块与块之间的重合大小为overlap，
            # 存储的为block_range范围内的字符
        # block_range：[(0, 300), (285, 585)]
        ber[i][0] = blocks[i]#  把分块好的训练数据存储到第一列上，即ber的第一列存储的为block_range范围内的分词
        ber[i][1].extend(es)# 第二列存储的为block_range范围内字符的实体(entity_type, nstart, nend, entity_str)     实体的start都是以每一块的第一个单词为参考

    for r, e1i, e2i in relations:# relations中的实体id是按ground truth行进行编号，从0开始
        if e1i not in e_dict or e2i not in e_dict:
            print("REIA lost due to sliding window")
            continue
        i1s, i2s = e_dict[e1i], e_dict[e2i]# i1s,i2s表示的是实体属于哪一个block
        intersec = set.intersection(set(i1s), set(i2s))# set.intersection(set1, set2) 求出集合1 和集合2 的交集：集合的交集为{0}，此集合有元素0，因此不属于空集合
        if intersec:# 两个实体在同一个block中
            for i in intersec:# entities[eli][0]:第45个实体的实体类型 entities[eli][1]：第45个实体的实体start id    block_range：[(0, 300), (285, 585), (570, 870), (855, 1155)]
                t1, s1, e1, es1 = entities[e1i][0], entities[e1i][1] - \
                    block_range[i][0], entities[e1i][2] - \
                    block_range[i][0], entities[e1i][3]
                t2, s2, e2, es2 = entities[e2i][0], entities[e2i][1] - \
                    block_range[i][0], entities[e2i][2] - \
                    block_range[i][0], entities[e2i][3]
                ber[i][2].append((r, (t1, s1, e1, es1), (t2, s2, e2, es2)))# ber = [[[第一块分词], [第一块中的实体距离第一个单词的距离],[关系, 实体1在block中的位置即单词距离块的第一个单词的位置,]],  [第二块], []]
        else:
            print("The two entities of the relationship are not on the same sentence\n")
    return ber


def get_question(question_templates, head_entity, relation_type=None, end_entity_type=None):
    """
    entities = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
    question_templates =
    k: get_question(question_templates, k) for k in entities
    Args:
        head_entity: (entity_type,start_idx,end_idx,entity_string) or entity_type
    """
    if relation_type == None:# 第一轮抽取头实体类型的问题
        question = question_templates['qa_turn1'][head_entity[0]] if isinstance( #isinstance(object, classinfo)判断obj是不是已知的数据类型
            head_entity, tuple) else question_templates['qa_turn1'][head_entity]
        # question_templates = {
        # "qu_turn1:{"FAC": "find all facility entities  in the context.",...}
        # "qu_turn2":{"('FAC', 'ART', 'FAC')": "find all facility entities in the context that have an artifact relationship with facility entity XXX.",,...}
        #                       }
    else:# end_entity_type = "GPE"    head_entity_type = ("LOC", 2, 3, "heartland")           relation_type = 'GPE-AFF'
        question = question_templates['qa_turn2'][str(
            (head_entity[0], relation_type, end_entity_type))]
        question = question.replace('XXX', head_entity[3])# 'find all geo political entities in the context that have a geo political affiliation relationship with location entity heartland.'
    return question


def block2qas(ber, dataset_tag, title="", threshold=1, max_distance=45):
    """
    Args:
        ber: (block,entities,relations)# block_er是把数据集中的一个文件的文本内容分为四块，其中ber是其中一块,ber(第几块中的分词, 实体， 关系)
        dataset_tag: type of dataset
        title: title corresponding to the passage to which the block belongs与块相对应的文章的标题
        threshold: only consider relationships where the frequency is greater than or equal to the threshold
        max_distance: used to filter relations by distance from the head entity用于通过头实体的距离过滤关系
    """
    if dataset_tag.lower() == "ace2004":
        entities = ace2004_entities# ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
        relations = ace2004_relations# ['ART', 'EMP-ORG','GPE-AFF', 'OTHER-AFF', 'PER-SOC', 'PHYS']
        idx1s = ace2004_idx1# idx1s = {'FAC': 0, 'GPE': 1, 'LOC': 2, 'ORG': 3, 'PER': 4, 'VEH': 5, 'WEA': 6}
        idx2s = ace2004_idx2# idx2s = {('ART', 'FAC'): 0, ('ART', 'GPE'): 1, ('ART', 'LOC'): 2, ('ART', 'ORG'): 3, ...}（关系类型, 尾实体）对应的0——41
        dist = ace2004_dist
        question_templates = ace2004_question_templates
    elif dataset_tag.lower() == 'ace2005':
        entities = ace2005_entities
        relations = ace2005_relations
        idx1s = ace2005_idx1
        idx2s = ace2005_idx2
        dist = ace2005_dist
        question_templates = ace2005_question_templates
    else:
        raise Exception("this data set is not yet supported")
    block, ents, relas = ber# block_er是把数据集中的一个文件的文本内容分为四块，其中ber是其中一块,ber(第几块中的分词, 实体， 关系)
    # block = ['like', 'many', 'heartland', 'states', ',', 'iowa', 'has', ..., ]
    # ents = [('GPE', 198, 199, 'america'), ('GPE', 205, 206, 'america'), ('GPE', 65, 66, 'usa'),...]距离每一块中第一个单词的位置
    # relas = [
    #     ('PHYS', ('GPE', 3, 4, 'states'), ('LOC', 2, 3, 'heartland')),
    #     ('PHYS', ('GPE', 32, 33, 'its'), ('LOC', 33, 34, 'borders')),
    #     ('PHYS', ('GPE', 19, 20, 'state'), ('LOC', 20, 21, 'lines')),
    #     ('EMP-ORG', ('PER', 41, 45, 'jim sciutto'), ('ORG', 38, 39, 'abc')),
    #       ...
    #          ]
    res = {'context': block, 'title': title}
   # "context" = ['like', 'many', 'heartland', 'states', ',', 'iowa', 'has', ..., ]
   # "title" = ['abc', 'news', 'story']
    # 给每一块加上title
    """
    turn1:找到所有实体
    turn2:找到实体之间的关系，实体之间的排列组合的关系
    """
   # QA turn 1
    dict1 = {k: get_question(question_templates, k) for k in entities}# entities = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
    # dict1 = {'FAC': 'find all facility entities in the context.',
    # 'GPE': 'find all geo political entities in the context.',
    # 'LOC': 'find all location entities in the context.',
    # 'ORG': 'find all organization entities in the context.',
    # 'PER': 'find all person entities in the context.',
    # 'VEH': 'find all vehicle entities in the context.',
    # 'WEA': 'find all weapon entities in the context.'}
    qat1 = {dict1[k]: [] for k in dict1}
    # {'find all facility entities in the context.': [],
    # 'find all geo political entities in the context.': [],
    # 'find all location entities in the context.': [],
    # 'find all organization entities in the context.': [],
    # 'find all person entities in the context.': [],
    # 'find all vehicle entities in the context.': [],
    # 'find all weapon entities in the context.': []
    #   }
    for en in ents: # ents = [('GPE', 198, 199, 'america'), ('GPE', 205, 206, 'america'), ('GPE', 65, 66, 'usa'),...]
        q = dict1[en[0]] # q = 'find all geo political entities  in the context.'
        qat1[q].append(en)# qat1的value是一个列表，因此是append()
    # qat1 = {
        # {'find all facility entities in the context.': [],
        # 'find all geo political entities in the context.': [[('GPE', 198, 199, 'america')]],
        # 'find all location entities in the context.': [],
        # 'find all organization entities in the context.': [],
        # 'find all person entities in the context.': [],
        # 'find all vehicle entities in the context.': [],
        # 'find all weapon entities in the context.': []
    #  }

    # QA turn 2
    if max_distance > 0:
        dict2 = {(rel[1], rel[0], rel[2][0]): [] for rel in relas}
        # relas: [
        #    [('PHYS', ('GPE', 3, 4, 'states'), ('LOC', 2, 3, 'heartland')),
        #    ('PHYS', ('GPE', 32, 33, 'its'), ('LOC', 33, 34, 'borders')),
        #    ('PHYS', ('GPE', 19, 20, 'state'), ('LOC', 20, 21, 'lines')),

        #       ...
        #          ]
                    # 把[[(关系类型), (实体1), (实体2)], [] , [] , ... ,[]] ——> {((实体1), 关系类型, 实体2类型):[(实体2), [], []]}   即把找尾个实体的问题转移到找到1实体已确定，找与实体1有某种关系的具有某个实体类型的实体2
        # dict2 = {
        #   (('GPE', 3, 4, 'states'), 'PHYS', 'LOC'): [],
        #   (('GPE', 32, 33, 'its'), 'PHYS', 'LOC'): [],
        #   (('GPE', 19, 20, 'state'), 'PHYS', 'LOC'): [],
        #   (('PER', 41, 45, 'jim sciutto'), 'EMP-ORG', 'ORG'): [],
        #   ...
        # }

        for rel in relas:
            dict2[(rel[1], rel[0], rel[2][0])].append(rel[2])
            # dict2 = {
            #   (('GPE', 3, 4, 'states'), 'PHYS', 'LOC'): [('LOC', 2, 3, 'heartland')],
            #   (('GPE', 32, 33, 'its'), 'PHYS', 'LOC'): [('LOC', 33, 34, 'borders')],
            #   (('GPE', 19, 20, 'state'), 'PHYS', 'LOC'): [('LOC', 20, 21, 'lines')],
            #   (('PER', 41, 45, 'jim sciutto'), 'EMP-ORG', 'ORG'): [('ORG', 38, 39, 'abc')],
            #   ...
            # }
        qat2 = []
        # ents = [('GPE', 198, 199, 'america'), ('GPE', 205, 206, 'america'), ('GPE', 65, 66, 'usa'),...]
        ents1 = sorted(ents, key=lambda x: x[1]) # key=lambda x: x[1] 为对前面的对象中的第二维数据的值进行排序
        # ents1即把每个实体在每一块中的相对位置进行排序 ents1 = [('LOC', 2, 3, 'heartland'), ('GPE', 3, 4, 'states'), ('GPE', 5, 6, 'iowa'), ('PER', 11, 12, 'people'),...
        '''
            通过max_distance先挑选出(头实体, 关系类型, 尾实体)，然后提出问题：找到所有和已知的头实体有rel_type关系的尾实体类型为end_type的实体
                ACE中有7中实体，实体之间可能存在的关系为42种
        '''
        for i, ent1 in enumerate(ents1):# i的初始化值为0 ,ent1表示头实体(实体类型, start, end, 实体)
            start = ent1[1]
            qas = {}
            for j, ent2 in enumerate(ents1[i+1:], i+1):# j的初始值为1(enmuerate(obj, k)从下标为k的位置开始遍历)，即忽略了第一个实体ents1[1:]
                # 双重for，以第一个实体为头实体，与ber中的其余实体进行比对
                if ent2[1] > start+max_distance:#  如果两个实体之间的距离大于max_distance，那么结束内循环
                    break
                else:
                    head_type, end_type = ent1[0], ent2[0]      # 头实体是一块数据中的第一个，尾实体是第一块数据中的除了第一个的其他的实体并且距离要小于max_distance
                    for rel_type in relations:# relations = ['ART', 'EMP-ORG','GPE-AFF', 'OTHER-AFF', 'PER-SOC', 'PHYS']
                        idx1, idx2 = idx1s[head_type], idx2s[(
                            rel_type, end_type)]
                        # idx1s = {'FAC': 0, 'GPE': 1, 'LOC': 2,...,} 头实体类型对应的为0——6
                        # idx2s = {('ART', 'FAC'): 0,('ART', 'GPE'): 1,...,('PHYS', 'WEA'): 41} （关系类型, 尾实体）对应的0——41 表示关系和尾实体类型，每个编号表示关系和尾实体对应，即1表示尾实体类型为“GPE”对应的关系为“ART”
                        if dist[idx1][idx2] >= threshold:# >=1 dist表示头实体类型分别为{'FAC': 0, 'GPE': 1, 'LOC': 2,...,}和尾实体类型为{'FAC': 0, 'GPE': 1, 'LOC': 2,...,}存在关系的为['ART', 'EMP-ORG','GPE-AFF', 'OTHER-AFF', 'PER-SOC', 'PHYS']的统计，因此dist为7行，每一行里的元素为42种关系，表示实体类型为FAC对应的尾实体类型和关系的统计
                            k = (ent1, rel_type, end_type)
                            q = get_question(# 已知(头实体, 关系类型, 尾实体类型)，进行第二轮问答
                                question_templates, ent1, rel_type, end_type)# 'find all geo political entities in the context that have a geo political affiliation relationship with location entity heartland.'
                            qas[q] = dict2.get(k, [])# 如果源字典dict2中有关键字k,那么输出其对应的value，如果没有k,那么输出为[]
                            # qas[q] = {问题:[],...,}
            qat2.append({"head_entity": ent1, "qas": qas})# [{'head_entity': ('LOC', 2, 3, 'heartland'), 'qas': {'find all geo political entities in the context that have a geo political affiliation relationship with location entity heartland.': [], 'find all geo political entities in the context that have a person or organization affiliation relationship with location entity heartland.': [], 'find all geo political entities in the context that have a physical relationship with location entity heartland.': [], 'find all person entities in the context that have a person or organization affiliation relationship with location entity heartland.': [], 'find all location entities in the context that have a physical relationship with location entity heartland.': [], 'find all organization entities in the context that have a employment, membership or subsidiary relationship with location entity heartland.': [], 'find all organization entities in the context that have a person or organization affiliation relationship with location entity heartland.': []}}]
    else:
        dict2 = {(rel[1], rel[0], rel[2][0]): [] for rel in relas}
        for rel in relas:
            dict2[(rel[1], rel[0], rel[2][0])].append(rel[2])
        qat2 = []
        for ent in ents:
            qas = {}
            for rel_type in relations:
                for ent_type in entities:
                    k = (ent, rel_type, ent_type)
                    idx1, idx2 = idx1s[ent[0]], idx2s[(rel_type, ent_type)]
                    if dist[idx1][idx2] >= threshold:
                        q = get_question(question_templates,
                                         ent, rel_type, ent_type)
                        qas[q] = dict2.get(k, [])
            qat2.append({'head_entity': ent, "qas": qas})
    qas = [qat1, qat2]
    res["qa_pairs"] = qas   # qas = [{找到所有类型性的实体，一共七个问题，字典的形式dict:[实体答案]}, [{第一个找的实体，是一个字典}, "qas":{第一个找到的实体与所有实体类型存在的可能出现的关系的问题，问题存不存在应该参照dist列表}]]
    return res # res =  {'context': block, 'title': title, "qa_pairs": [qat1：{第一轮抽取头实体的问题}, qat2第二轮抽取尾实体和关系类型的问题]}
# qas = [{'find all facility entities  in the context.': [],'find all geo political entities  in the context.': [['GPE', 0, 1, 'cuban'], ['GPE', 17, 18, 'nation'], ['GPE', 133, 134, 'cuba'], ['GPE', 136, 137, 'where'], ['GPE', 141, 142, 'country'], ['GPE', 152, 153, 'they'], ['GPE', 31, 32, 'rest']],。。。}, [{'head_entity': ['GPE', 0, 1, 'cuban'],
        # ['head_entity': ('LOC', 2, 3, 'heartland'), 'qas': {'find all geo political entities in the context that have a geo political affiliation relationship with location entity heartland.': [], 'find all geo political entities in the context that have a person or organization affiliation relationship with location entity heartland.': [], 'find all geo political entities in the context that have a physical relationship with location entity heartland.': [],]
def char_to_wordpiece(passage, entities, tokenizer):
    '''
    把一个word拆分成一片一片，把词的本身意思和时态分开，有效的减少词表的数量，提高训练速度
    Args:
        passage: like many heartland states , iowa has had trouble keeping young people down on the farm or anywhere within state lines .
                with population waning , the state is looking beyond its borders for newcomers .
                as abc ' s jim sciutto reports , one little town may provide a big lesson .
                原始文本分词后再合并，添加了一些空字符
        entities: [['GPE', 926, 933, 'america'], ['GPE', 957, 964, 'america'], ['GPE', 330, 333, 'usa'], ['PER', 367, 374, '2 , 000'], ['GPE', 359, 363, 'town'],
        tokenizer:
    实体不以字符的个数标识，以单词的个数标识
    Returns:

    '''
    entities1 = []
    tpassage = tokenizer.tokenize(passage)
    for ent in entities:
        ent_type, start, end, ent_str = ent
        s = tokenizer.tokenize(passage[:start])# 实体头之前的字符进行分词
        start1 = len(s)# 实体头前面有多少个字符存在，即实体头以word为单位的长度，而不是以字符级
        ent_str1 = tokenizer.tokenize(ent_str)
        end1 = start1 + len(ent_str1)
        ent_str2 = tokenizer.convert_tokens_to_string(ent_str1)
        assert tpassage[start1:end1] == ent_str1
        entities1.append((ent_type, start1, end1, ent_str2))
    return entities1


# 预训练函数
def process(data_dir, output_dir, tokenizer, is_test, window_size, overlap, dataset_tag, threshold=1, max_distance=45):
    """
    Args:
        data_dir: data directory ->/data/raw_data/ACE2004/train0
        output_dir: output directory
        tokenizer: tokenizer of pretrained model
        is_test: whether it is test data
        window_size: sliding window size滑动窗口size
        overlap: overlap between two adjacent windows两个相邻窗户之间的重叠
        dataset_tag: type of dataset
        threshold： only consider relationships where the frequency is greater than or equal to the threshold
            只考虑频率大于或等于阈值的关系
        max_distance: used to filter relations by distance from the head entity
            用于通过距离头实体的距离来过滤关系
    """
    ann_files = [] # 以ann为后缀的训练数据:['/home/wgy/paper/REIA-测试完预训练/data/raw_data/ACE2004/train0/VOA20001101.2100.3077.ann']
    txt_files = [] # 以txt为后缀的训练数据:['/home/wgy/paper/REIA-测试完预训练/data/raw_data/ACE2004/train0/NBC20001003.1830.0755.txt']
    data = []
    for f in os.listdir(data_dir): # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表，打开原始文件的train0下的文件名
        if f.endswith('.txt'): # 文件以txt结尾
            txt_files.append(os.path.join(data_dir, f))
        elif f.endswith('.ann'):
            ann_files.append(os.path.join(data_dir, f))
    ann_files = sorted(ann_files)# 把以ann为结尾的文件名放到列表中并排序 len(ann_files) = 279
    txt_files = sorted(txt_files)# 以txt结尾的文件名放到列表中，两个都是指训练数据
    for ann_path, txt_path in tqdm(zip(ann_files, txt_files), total=len(ann_files)):
        # zip(a, b)把两个文件进行合并，得到的结果为元组迭代器
        with open(txt_path, encoding='utf-8') as f:
            raw_txt = f.read()
            txt = [t for t in raw_txt.split('\n') if t.strip()]# 原始文本进行按段落分割开，即把每一行元素作为列表的一个元素，并删除每个元素的开头和结尾处的所有空白字符，不删除中间的空白字符
            # txt = [' ABC20001001.1830.0973 ', ' NEWS STORY ', ' 10/01/2000 18:46:13.59 ',  "like many heartlan。。。", "on homecoming 。。。" , ... ,]

        # get the title information, the title will be added to all windows of a passage
        title = re.search('[A-Za-z_]+[A-Za-z]', txt[0]                # re.search('[A-Za-z_]+[A-Za-z]', txt[0]) <re.Match object; span=(1, 4), match='ABC'>
                          ).group().split('-') + txt[1].strip().split() # re.search('[A-Za-z_]+[A-Za-z]', txt[0]).group()——> "ABC"
        # title = ['ABC', 'NEWS', 'STORY']                            # re.search('[A-Za-z_]+[A-Za-z]', txt[0]).group().split("-")——> ["ABC"]
        # 字符串转化为列表 "ABC".split()——> ["ABC"]
        title = " ".join(title)# 字符串的拼接title = "ABC NEWS STORY"           "-".join(title)——>"ABC-NEWS_STORY"
        title = tokenizer.tokenize(title) # 对文本进行 tokenization（分词）之后，返回的分词的 token 词, 并自动转换为小写,结果为列表
        # title= ['abc, 'news', 'story']

        ntxt = ' '.join(txt[3:])# 把正文字符串通过空格进行拼接，即原来的正文构成了一个列表，列表的每一个元素表示每一个段落，现在将各个段落通过空格连接到一起
        # len(raw_txt) = 2286 len(ntxt) = 2205
        ntxt1 = tokenizer.tokenize(ntxt)
        # ntxt1: ['like', 'many', 'heartland', 'states', ',', 'iowa', 'has', 'had', 'trouble', 'keeping', 'young', 'people', 'down', 'on', 'the', 'farm', 'or', 'anywhere', 'within', 'state', 'lines', '.'
        ntxt2 = tokenizer.convert_tokens_to_string(ntxt1)# 把分词转化为字符串，convert_tokens_to_ids把分词转化为在词表中的id
        # ntxt2: like many heartland states , iowa has had trouble keeping young people down on the farm or anywhere within state lines .
        offset = raw_txt.index(txt[3])# offset：65
        # 去掉文件头的字符串的offset，即正文相对于原始文本的偏移

        entities, relations = aligment_ann(# ann对齐ground truth，对齐关系：实体头和尾是出现在ground truth第几行，即relation( 关系类型，头实体在哪一行，尾实体在哪一行)              entities对齐的是ntxt2的位置
            raw_txt[offset:], ntxt2, ann_path, offset)               # entities = [['GPE', 926, 933, 'america'], ['GPE', 957, 964, 'america'], ['GPE', 330, 333, 'usa'], ['PER', 367, 374, '2 , 000']。。]  relations =  [['PHYS', 44, 84], ['PHYS', 90, 94], ['PHYS', 88, 93], ['EMP-ORG', 99, 97], ['EMP-ORG', 98, 96]...]
        # entities: （实体类型, 实体的跨度start, 实体的跨度end, 实体）       这里的span start是ntxt2的下标位置
        # relations: (关系类型, 实体头属于第几个实体， 实体尾属于第几个实体) 实体头和尾表示ground truth中第几行实体
        # convert entitiy index from char level to wordpiece level
        entities = char_to_wordpiece(ntxt2, entities, tokenizer)# 实体index从word级转换为wordpiece级[('GPE', 198, 199, 'america'), ('GPE', 205, 206, 'america'), ('GPE', 65, 66, 'usa'), ('PER', 74, 77, '2 , 000'), ('GPE', 72, 73, 'town'), ]
        if is_test:# 预训练期间，没有调用parser中的is_test, 因此activate_true没有被调用，is_test则默认为False
            data.append({"title": title, "passage": ntxt1,
                         "entities": entities, "relations": relations})
        else:
            block_er = get_block_er(# block_er = ber = [[[第一块分词], [第一块中的实体距离第一个单词的距离],[关系, 头实体在block中的位置即单词距离块的第一个单词的位置]],  [第二块], []]       s1:头实体在那一block中的位置即块的相对位置
                ntxt1, entities, relations, window_size, overlap, tokenizer)# ntxt1 = 数据集正文分词后的列表  entities = 实体是按照词级进行划分的，即第一个实体相对于tokenize的单词数
            # block_er存储了数据集中一个文件的内容，通过window_size,overlap一共分为了四块，
            #       即数据集其中一个文件被分为了四块，每一块的字符的长度为300，overlap为15
            # block_er[0] = [
            #   [like', 'many', 'heartland', 'states', ',', 'iowa', 'has', 'had', 'trouble', 'keeping', ... ,]
            #   [('GPE', 198, 199, 'america'), ('GPE', 205, 206, 'america'), ('GPE', 65, 66, 'usa'), ('PER', 74, 77, '2 , 000'),.....]
            #   [('PHYS', ('GPE', 3, 4, 'states'), ('LOC', 2, 3, 'heartland')), ('PHYS', ('GPE', 32, 33, 'its'), ('LOC', 33, 34, 'borders')), ('PHYS', ('GPE', 19, 20, 'state'), ('LOC', 20, 21, 'lines')),...]
            #               ]
            # block_er[0][0] = ['like', 'many', 'heartland', 'states', ',', 'iowa', 'has', 'had', 'trouble', 'keeping', ... ,....]
            for ber in block_er:
                # block_er是把数据集中的一个文件的文本内容分为四块，其中ber是其中一块,ber(第几块, 实体， 关系)
                data.append(block2qas(ber, dataset_tag,
                                      title, threshold, max_distance))# data = # res =  {'context': block, 'title': title, "qa_pairs": [qat1第一轮抽取头实体的问题, qat2第二轮抽取尾实体和关系类型的问题]}
                 # data = {'context': block, 'title': title, "qa_pairs": qas}
                # data中存放的是把训练集中的一个文件分为len(passage) / 300 块，并把提出两轮问题，第一轮找到所有实体，第二轮以第一轮找到的所有实体作为头实体，与其余的关系和实体进行匹配
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        # output_dir = '/home/wgy/paper/REIA/data/cleaned_data/ACE2004/bert-base-uncased_overlap_15_window_300_threshold_1_max_distance_45'
        # data_dir = "./data/raw_data/ACE2004/train0 "
    # join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串
    save_path = os.path.join(output_dir, os.path.split(data_dir)[-1]+".json")
    print("save_path = {}".format(save_path))
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='./data/raw_data/ACE2005/train')
    parser.add_argument(
        "--dataset_tag", choices=["ace2004", 'ace2005'], default='ace2005')
    parser.add_argument("--window_size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=15)
    parser.add_argument("--is_test", action="store_true")
    # store_true表示触发时为ture，在脚本中调用参数时，为True
    """
    在demo.py中指定action = "store_true"时：
    parser.add_argument('--is_test',action='store_true')
    运行时：
    python demo.py 默认是False
    python demo.py --is_test 默认是TrueG
        相当于“开关”作用
    """
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--output_base_dir",
                        default="./data/cleaned_data/ACE2005")
    parser.add_argument("--pretrained_model_path",
                        default='./bert-base-uncased')
    parser.add_argument("--max_distance", type=int, default=45,
                        help="used to filter relations by distance from the head entity")
    args = parser.parse_args()
    if not args.is_test:
        output_dir = "{}/{}_overlap_{}_window_{}_threshold_{}_max_distance_{}".format(args.output_base_dir, os.path.split(
            args.pretrained_model_path)[-1], args.overlap, args.window_size, args.threshold, args.max_distance)
    else:
        output_dir = args.output_base_dir
    # output_dir = '/home/wgy/paper/REIA/data/cleaned_data/ACE2004/bert-base-uncased_overlap_15_window_300_threshold_1_max_distance_45'
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    process(args.data_dir, output_dir, tokenizer, args.is_test,# data_dir = /data/raw_data/ACE2004/train0
            args.window_size, args.overlap, args.dataset_tag, args.threshold, args.max_distance)