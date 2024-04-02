import torch
def iou(box,boxes,isMin=False):
    # box是置信度最高的框，boxes的其他的没它高的所有框，box与boxes计算iou
    # 如果iou过阈值，那么就是识别的同一个物体，去掉
    #box (x1,y1,x2,y2)

    #面积
    box_area=(box[2]-box[0])*(box[3]-box[1])
    boxes_areas=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])

    #交集
    xx1=torch.maximum(box[0],boxes[:,0])
    yy1=torch.maximum(box[1],boxes[:,1])
    xx2=torch.minimum(box[2],boxes[:,2])
    yy2=torch.minimum(box[3],boxes[:,3])

    w=torch.maximum(torch.Tensor([0]),xx2-xx1)
    h=torch.maximum(torch.Tensor([0]),yy2-yy1)

    ovr_area=w*h

    if isMin:
        return ovr_area/torch.min(box_area,boxes_areas)
    else:
        return ovr_area/(box_area+boxes_areas-ovr_area)

def nms(boxes,thresh=0.3,isMin=False):
    #boxes的每一个box的第0个位置就是置信度，先来排个序,降序的
    new_boxes=boxes[boxes[:,0].argsort(descending=True)]
    keep_boxes=[]  # 要保留的框
    while len(new_boxes)>0:
        _box=new_boxes[0]
        keep_boxes.append(_box)
        if len(new_boxes)>1:
            _boxes=new_boxes[1:]
            new_boxes=_boxes[torch.where(iou(_box,_boxes,isMin)<thresh)]
        else:
            break

    return torch.stack(keep_boxes)

if __name__=="__main__":
    # box=torch.Tensor([0,0,4,4])
    # boxes=torch.Tensor([[4,4,5,5],[1,1,5,5]])
    # print(iou(box,boxes))
    boxes=torch.tensor([[0.5,1.5,1.5,10,10],[0.9,2,2,11,11],[0.4,8,8,12,12]])
    print(nms(boxes,0))