import numpy as np
improt pygame

# this is for time

# this is for a list of minute/sec price updates per day. Each day is a level
def step_time_minute_price_updates(fps, coord1, coord2, t1, t2):
    x1, y1 = coord1
    x2, y2 = coord2
    dx = x2-x1
    dy = y2-y1
    division_fps = abs(t2-t1)*fps
    step_x, step_y = (dx/division_fps, dy/division_fps)
    
    return step_x, step_y

# this is for a list of minute/sec price updates per day. Each day is a level
def step_time_list_minute_price_updates(fps, object, lst_coord, lst_timepoints):
    i = 0
    while j<len(lst_coord):
        j = i+1
        coord1 = lst_coord(i)
        coord2 = lst_coord(j)
        t1 = lst_timepoints[i]
        t2 = lst_timepoints[j]
        step_x, step_y = step_time_minute_price_updates(fps, coord1, coord2, t1, t2)
        object.set_position(object.x + step_x, object.y + step_y)

        i+=1

def step_time_daily_price(fps, object, open_price, high_price, low_price, close_price):
    lst_price = [open_price, high_price, low_price, close_price]
    x1, y1 = open_price
    x2, y2 = high_price
    x3, y3 = low_price
    x4, y4 = close_price

    i = 0
    while j<5:
        j = i+1
        x1, y1 = lst_price[i]
        x2, y2 = lst_price[j]
        dx = x2-x1
        dy = y2-y1

        step_x, step_y = (dx / (fps*4), dy / (fps*4))
        object.set_position(object.x + step_x, object.y + step_y)


