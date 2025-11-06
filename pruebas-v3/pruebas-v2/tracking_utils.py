# tracking_utils.py
import math

def update_tracks(tracked_objs, detections, max_dist=35, max_miss=10, next_id_start=None):
    """
    tracked_objs: dict id -> {'pt': (x,y), 'miss': int}
    detections: list[(x,y)]
    devuelve: tracked_objs_actualizado, next_id
    """
    if next_id_start is None:
        if tracked_objs:
            next_id = max(tracked_objs.keys()) + 1
        else:
            next_id = 1
    else:
        next_id = next_id_start

    # marcar todos como no actualizados
    for oid in list(tracked_objs.keys()):
        tracked_objs[oid]['updated'] = False

    # asociar detecciones
    for (cx, cy) in detections:
        best_oid = None
        best_dist = 1e9
        for oid, data in tracked_objs.items():
            px, py = data['pt']
            dist = math.hypot(cx - px, cy - py)
            if dist < best_dist:
                best_dist = dist
                best_oid = oid
        if best_oid is not None and best_dist < max_dist:
            tracked_objs[best_oid]['pt'] = (cx, cy)
            tracked_objs[best_oid]['miss'] = 0
            tracked_objs[best_oid]['updated'] = True
        else:
            tracked_objs[next_id] = {
                'pt': (cx, cy),
                'miss': 0,
                'updated': True,
            }
            next_id += 1

    # borrar los que lleven mucho sin verse
    for oid in list(tracked_objs.keys()):
        if not tracked_objs[oid].get('updated', False):
            tracked_objs[oid]['miss'] += 1
        if tracked_objs[oid]['miss'] > max_miss:
            del tracked_objs[oid]

    return tracked_objs, next_id
