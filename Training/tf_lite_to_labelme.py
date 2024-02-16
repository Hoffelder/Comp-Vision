import labelme 
import base64

data       = labelme.LabelFile.load_image_file("Solida/IMG_4001.JPG")
image_data = base64.b64encode(data).decode('utf-8')

print(image_data)

x0,y0 = np.array(shapes["points"])[0]
x1,y1 = np.array(shapes["points"])[1]
points.append([x0/w, y0/h,  x1/w, y0/h, x1/w, y1/h, x0/w, y1/h])


y = (bb[[1,3,5,7]]*h).astype(int)
x = (bb[[0,2,4,6]]*w).astype(int)

x0,y0,x1,y1 = x[0],y[0],x[1],y[2]

new_point = {
            "label": "solida",
            "points": [
                [
                    int(x0*w), 
                    int(y0*h)
                ],
                [
                    int(x1*w), 
                    int(y1*h)
                ]
            ],
            "group_id": null,
            "shape_type": "rectangle",
            "flags": {}
            }



{
    "version": "5.0.1",
    "flags": {},
    "shapes": [
        new_point,
        new_point
    ],
    "imagePath":   "IMG_4003.JPG",
    "imageData":   image_data,
    "imageHeight": 3024,
    "imageWidth":  4032
}