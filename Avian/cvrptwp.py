import os
import json
from datetime import datetime, timedelta
import sys
import uuid
from utils.config import Config
from utils.db import getGroupVehicleByGroupId, getListOrderGoodsByOrderId, getOrderByTaskId, getOrderGoodsByOrderId, getStoreByCode, query_db
from utils.dto import GroupVehicle, Order, OrderGoods, Plan, Route, RouteOrder, Store
from utils.enum import PlanStatus
from utils.gql import updateProgress
from utils.utils import setError
import folium
from folium import plugins
from dateutil import tz
from utils.vrp.time_utils import validate_time_window, convert_time_to_minutes
from utils.vrp.clustering import create_balanced_cluster_locations
from utils.vrp.routing import solve_pickup_delivery_vrp
from utils.vrp.store_utils import getStoreOpenTime
from utils.vrp.map_utils import getTableMatrix, get_route_directions
from math import radians, cos, sin, asin, sqrt
# import psutil
# import signal

# def check_memory_usage():
#     process = psutil.Process()
#     memory_info = process.memory_info()
#     # print(memory_info)
#     if memory_info.rss > 1000000000:  # 1G
#         raise MemoryError("Memory usage too high")
# class timeout:
#     def __init__(self, seconds=10):
#         self.seconds = seconds
#     def __enter__(self):
#         signal.signal(signal.SIGALRM, self.handle_timeout)
#         signal.alarm(self.seconds)
#     def __exit__(self, *args):
#         signal.alarm(0)
#     def handle_timeout(self, *args):
#         raise TimeoutError("Function timed out")
    
def cvrptwp(item: Plan, isPreview: bool):
    # Add global set to track unallocated locations
    global_unallocated_locations = set()
    unallocated = []  # Initialize unallocated list
    def add_to_unallocated(location, orders, demand, time_window, reason, vehicle_code=None):
        if location not in global_unallocated_locations:
            global_unallocated_locations.add(location)
            unallocated.append({
                'location': location,
                'demand': demand,
                'orders': orders,
                'time_window': time_window,
                'vehicle': vehicle_code,
                'reason': reason
            })
    
    # Configuration settings
    configPlan = {
        # 'use_all_vehicles': False,         # Try to utilize all available vehicles
        'use_all_vehicles': item.useAllVehicle,         # Try to utilize all available vehicles
        'allow_overtime': False,            # Allow overtime beyond vehicle end time
        'overtime_limit': 60,              # Maximum overtime minutes allowed
        'cluster_strategy': 'balanced',    # balanced/capacity/time/distance/fairness based
        'debug_output': Config.DEBUG == '1',         # Show detailed output in preview mode
        'preparation_time': 2,           # Vehicle preparation time at depot
        'customer_service_time': {
            'base': 10,                   # Base service time per stop
            'per_order': 0.5,               # Additional minutes per order
        },
        'map_option': "osrm" if Config.DEBUG == '1' else item.mode, # googlemaps/osrm
    }
    
    # Validate and adjust config based on business rules
    if not configPlan['allow_overtime']:
        configPlan['overtime_limit'] = 0

    if configPlan['debug_output']:
        print("\nConfiguration Settings:")
        print("-" * 50)
        for key, value in configPlan.items():
            print(f"{key}: {value}")
        print("-" * 50)
    # print(item)
    '''
    Set HomeBase LatLon
    '''
    homeBase = None
    if item.brancLat is not None and item.brancLon is not None :
        homeBase = {"lat": item.brancLat, "lon": item.brancLon}

    if not homeBase:
        return setError(item, ['No home base location configured'])
    
    '''
    Get Orders By TaskId
    '''
    listOrder = getOrderByTaskId(item.taskId)
    if not listOrder:
        setError(item, ['The order has not been configured'])
        return None

    '''
    Get List Configured Vehicle
    '''
    listGroupVehicle = getGroupVehicleByGroupId(item.groupId)
    if not listGroupVehicle:
        setError(item, ['The vehicle has not been configured'])
        return None

    '''
    Grouping Orders By Store / LatLon
    '''
    
    # Fetch store information for active stores
    listStore = getStoreByCode({order.get('storeCode') for order in listOrder if order.get('storeCode')})
    store_codes = set()
    orders_by_location = {}
    store_assignments = {}  # Track store-to-vehicle assignments
    closed_stores = {}  # Track closed stores and their orders

    # Group orders by their geographic coordinates and collect store codes
    for raw_order in listOrder:
        order = Order.from_dict(raw_order)
        store_codes.add(order.storeCode)
        location_key = f"{order.lon},{order.lat}"
        
        # Check if store is open with enhanced validation
        store_time = getStoreOpenTime(listStore=listStore, order=order)
        if not store_time:
            # Store is closed - add to closed stores tracking
            if location_key not in closed_stores:
                closed_stores[location_key] = {
                    'orders': [],
                    'total_volume': 0,
                    'total_weight': 0,
                    'store_code': order.storeCode
                }
            closed_stores[location_key]['orders'].append(raw_order)
            closed_stores[location_key]['total_volume'] += order.totalVolume * 100000
            closed_stores[location_key]['total_weight'] += order.totalWeight
            continue
            
        # Validate store hours and time window
        store_open, store_close = store_time
        if store_close <= store_open:
            # Invalid store hours - add to closed stores tracking
            if location_key not in closed_stores:
                closed_stores[location_key] = {
                    'orders': [],
                    'total_volume': 0,
                    'total_weight': 0,
                    'store_code': order.storeCode
                }
            closed_stores[location_key]['orders'].append(raw_order)
            closed_stores[location_key]['total_volume'] += order.totalVolume * 100000
            closed_stores[location_key]['total_weight'] += order.totalWeight
            continue
        
        # Process valid store/order
        if location_key not in orders_by_location:
            location_data = {
                'orders': [raw_order],
                'total_volume': order.totalVolume * 100000,
                'total_weight': order.totalWeight,
                'time_window': store_time
            }
        else:
            location_data = orders_by_location[location_key]
            location_data['orders'].append(raw_order)
            location_data['total_volume'] += order.totalVolume * 100000
            location_data['total_weight'] += order.totalWeight
            location_data['time_window'] = store_time
        
        orders_by_location[location_key] = location_data

    # Process all closed stores and add their orders to unallocated
    for location_key, store_data in closed_stores.items():
        add_to_unallocated(
            location=location_key,
            demand=store_data['total_volume'],
            orders=store_data['orders'],
            time_window=None,  # No valid time window for closed stores
            reason=f'Store {store_data["store_code"]} is closed or has invalid operating hours'
        )

    if not orders_by_location:
        return setError(item, ['No valid delivery locations found'])
        
    print(f"Total locations: {len(orders_by_location)}")
    print(f"Closed stores: {len(closed_stores)}")
    print(f"Unallocated orders: {len(unallocated)}")
    '''
    Get List order goods
    '''
    # Extract order IDs from listOrderLatLon
    orderIds = []
    
    for orders in orders_by_location.values():
        for order in orders['orders']:
            orderIds.append(order['id'])
    
    # Get order goods using the collected IDs
    listOrderGoods = getListOrderGoodsByOrderId(orderIds)
    
    if listOrderGoods is None:
        return setError(item, ['No Order Goods'])

    '''
    Create Distance and Time Matrix
    '''
    latLonKeys = list(orders_by_location.keys())
    if not latLonKeys:
        return setError(item, ['No valid delivery locations found'])
        
    latLonKeys.insert(0, f"{homeBase['lon']},{homeBase['lat']}")  # type: ignore
    
    table_matrix = getTableMatrix(latLonKeys)
    if table_matrix is None:
        return setError(item, ['Failed get table matrix'])
    
    distance_matrix = table_matrix.get('distances')
    # Convert distance_matrix float to int
    if distance_matrix is None:
        return setError(item, ['Failed get distance matrix'])
    distance_matrix = [[int(distance) for distance in row] for row in distance_matrix]
    
    time_matrix = table_matrix.get('durations')
    # Convert time_matrix float to int
    if time_matrix is None:
        return setError(item, ['Failed get time matrix'])
    time_matrix = [[int(duration / 60) for duration in row] for row in time_matrix]
    
    depot_start = convert_time_to_minutes("08:00")
    depot_end = convert_time_to_minutes("20:59")
    
    if depot_start is None or depot_end is None:
        return setError(item, ['Invalid depot time window'])
    
    time_windows = [orders_by_location[latLonKey]['time_window'] for latLonKey in orders_by_location.keys()]
    
    time_windows.insert(0, (depot_start,depot_end))
    
    if len(latLonKeys) < 2:  # Only depot exists
        return setError(item, ['No valid delivery locations available'])

    vehicle_capacities = [int(GroupVehicle.from_dict(gVehicle).maxCapacity * 100000) for gVehicle in listGroupVehicle]

    total_volumes = [orders_by_location[latLonKey]['total_volume'] for latLonKey in orders_by_location.keys()]
    total_volumes.insert(0,0)

    data = {
        'distance_matrix': distance_matrix,
        'time_matrix': time_matrix,
        'time_windows': time_windows,
        'demandLB': total_volumes,
        'vehicle_capacitiesLB': vehicle_capacities,
        'depot': 0,
    }
    
    # Create vehicle-specific time windows with validation
    vehicle_time_windows = []

    for gVehicle in listGroupVehicle:
        vehicle = GroupVehicle.from_dict(gVehicle)
        start_time = convert_time_to_minutes(vehicle.startTime) if vehicle.startTime else 480
        end_time = convert_time_to_minutes(vehicle.endTime) if vehicle.endTime else 1259
        
        if start_time is None or end_time is None:
            start_time, end_time = 480, 1020
        
        # Initialize assignment tracking
        vehicle_time_windows.append(validate_time_window(start_time, end_time))
        
    depot_location = latLonKeys[0]
    
    # For each vehicle, store start/end locations
    vehicle_locations = {}
    for gVehicle in listGroupVehicle:
        vehicle = GroupVehicle.from_dict(gVehicle)
        vehicle_locations[vehicle.id] = {
            'start': (vehicle.startLocationLat, vehicle.startLocationLon),
            'end': (vehicle.endLocationLat, vehicle.endLocationLon)
        }
        
    clustered_locations, clustered_demands, clustered_time_windows, cluster_unallocated, cluster_vehicle = create_balanced_cluster_locations(
        orders_by_location,
        data,
        vehicle_time_windows,
        depot_location,
        item,
        configPlan,
        listStore,
        listGroupVehicle
    )
    
    
    print(f"clustered_locations : {len(clustered_locations)}")
    
    for u in cluster_unallocated:
        unallocated.append(u)
        
    
    
    # Process each cluster with VRP
    all_routes = []
    all_route_vehicle = []
    
    setProgress(item=item,progress=40,msg='Calculating routes ..',logs=[])
    
    # print(f"depot_location : {depot_location}")
    
    # Process each cluster if we have valid clusters
    if clustered_locations and clustered_demands and clustered_time_windows and cluster_vehicle:
        for cluster_idx, (cluster_points, cluster_demands, cluster_time_windows, vehicle) in enumerate(
            zip(clustered_locations, clustered_demands, clustered_time_windows, cluster_vehicle)):
            
            # Add vehicle start/end locations to cluster points
            start_point = f"{vehicle_locations[vehicle.id]['start'][1]},{vehicle_locations[vehicle.id]['start'][0]}"
            end_point = f"{vehicle_locations[vehicle.id]['end'][1]},{vehicle_locations[vehicle.id]['end'][0]}"
            
            # Update matrices to include start/end points
            # Update distance_matrix and time_matrix accordingly
            cluster_distance_matrix = []
            cluster_time_matrix = []
            
            # Get indices for start/end points in original matrix
            point_indices = [latLonKeys.index(point) for point in cluster_points]
            
            diff_start = False
            diff_end = False

            # Check if start point is different from depot
            def haversine_meters(lon1, lat1, lon2, lat2):
                """Calculate distance between two lat/lon points in meters"""
                R = 6371000  # Earth's radius in meters
                lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
                dlon = lon2 - lon1 
                dlat = lat2 - lat1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a))
                return R * c

            # Check if start point is different from depot, with tolerance
            start_lon, start_lat = map(float, start_point.split(','))
            end_lon, end_lat = map(float, end_point.split(','))
            depot_lon, depot_lat = map(float, depot_location.split(','))
            distance_start = haversine_meters(start_lon, start_lat, depot_lon, depot_lat)
            distance_end = haversine_meters(end_lon, end_lat, depot_lon, depot_lat)
            distance_start_end = haversine_meters(end_lon, end_lat, start_lon, start_lat)

            vehicle.start_index = 0
            vehicle.end_index = 0
            
            # if distance_start > 15:  # More than 15m apart
            #     diff_start = True
            #     cluster_points.insert(0,start_point)
            #     cluster_demands.insert(0,0)
            #     cluster_time_windows.insert(0,(convert_time_to_minutes(vehicle.startTime) if vehicle.startTime else 480, convert_time_to_minutes(vehicle.endTime) if vehicle.endTime else 1259))
            #     vehicle.end_index = 1
                
            # if distance_start_end < 15:
            #     vehicle.end_index = 0
            # elif distance_end > 15: 
            #     diff_end = True
            #     cluster_points.append(end_point)
            #     cluster_demands.append(0)
            #     cluster_time_windows.append((convert_time_to_minutes(vehicle.startTime) if vehicle.startTime else 480, convert_time_to_minutes(vehicle.endTime) if vehicle.endTime else 1259))
            #     vehicle.end_index = len(cluster_points) -1
            
            new_cluster_matrix = {}
            
            if diff_start or diff_end:
                # Get matrix for start point and add it
                get_cluster_matrix = getTableMatrix([p for p in cluster_points])
                if get_cluster_matrix:
                    new_cluster_matrix = get_cluster_matrix

            # Create matrices with correct dimensions
            n = len(cluster_points)
            cluster_distance_matrix = [[0] * n for _ in range(n)]
            cluster_time_matrix = [[0] * n for _ in range(n)]

            # Fill matrices based on start/end configurations
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                        
                    if diff_start or diff_end:
                        # Use new matrix values if start/end points are different
                        n1 = new_cluster_matrix.get('distances')
                        n2 = new_cluster_matrix.get('durations')
                        if n1 and n2:
                            cluster_distance_matrix[i][j] = int(n1[i][j])
                            cluster_time_matrix[i][j] = int(n2[i][j] / 60)
                    else:
                        # Handle distances between main points
                        idx1 = point_indices[i] if i < len(point_indices) else 0
                        idx2 = point_indices[j] if j < len(point_indices) else 0
                        cluster_distance_matrix[i][j] = distance_matrix[idx1][idx2]
                        cluster_time_matrix[i][j] = time_matrix[idx1][idx2]
                        
            pickup_delivery_demands = []  
            # Inside the cvrptwp function where demands are processed:
            if diff_start:
                # If starting from non-depot location
                pickup_delivery_demands = [0]  # Start location has 0 demand
                
                # Add depot's pickup demand
                pickup_delivery_demands.append(sum(cluster_demands[2:]))  # Depot has total pickup demand
                
                # Add demands for delivery locations
                for demand in cluster_demands[2:]:
                    pickup_delivery_demands.append(-demand)  # Negative demand for deliveries
            else:
                # Standard case - starting from depot
                pickup_delivery_demands = [sum(cluster_demands[1:])]  # Depot has total pickup demand
                for demand in cluster_demands[1:]:
                    pickup_delivery_demands.append(-demand)  # Negative demand for deliveries
                    
            # print("-" * 50)
            # print(f"cluster_points : {cluster_points}")
            # print(f"cluster_points : {len(cluster_points)}")
            # # print(f"first cluster_points : {cluster_points[0]}")
            # print(f"start_point : {start_point}")
            # # print(f"distance_start : {distance_start}")
            # # print(f"end_point : {end_point}")
            # # print(f"distance_end : {distance_end}")
            # # print(f"cluster_time_windows : {cluster_time_windows}")
            # # print(f"cluster_distance_matrix : {len(cluster_distance_matrix)}")
            # # print(f"cluster_time_matrix : {len(cluster_time_matrix)}")
            # # print(f"pickup_delivery_demands : {len(pickup_delivery_demands)}")
            # # print(f"pickup_delivery_demands : {pickup_delivery_demands}")
            # print("-" * 50)
            # continue
            
            
            # try:
            #     with timeout(seconds=30):
            #         check_memory_usage()
            # Solve VRP for this cluster
            solution, cluster_manager, cluster_routing, dropped_nodes = solve_pickup_delivery_vrp(
                cluster_points,
                {'cluster_distance_matrix':cluster_distance_matrix,'cluster_time_matrix':cluster_time_matrix}, 
                cluster_time_windows,
                vehicle,
                pickup_delivery_demands,
                orders_by_location
            )
            # except (TimeoutError, MemoryError) as e:
            #     print(f"Resource error: {e}")
            #     return None, None, None, None
            
            # for d in dropped_nodes:
            #     print(f"dropped_nodes : {cluster_points[d]}")
            
            if solution:
                # print(f"solution : {solution}")
                # Extract solution
                time_dimension = cluster_routing.GetDimensionOrDie('Time')
                route = []
                route_load = 0
                route_distance = 0
                time_list = []
                time_matrix_list = []
                dropped_locations = []

                # Check for dropped locations
                for node in range(cluster_manager.GetNumberOfNodes()):
                    if cluster_routing.IsStart(node) or cluster_routing.IsEnd(node):
                        continue
                    if solution.Value(cluster_routing.NextVar(node)) == node:
                        node_idx = cluster_manager.IndexToNode(node)
                        dropped_locations.append(cluster_points[node_idx])
                        # Add to unallocated with reason
                        add_to_unallocated(
                            cluster_points[node_idx],
                            orders_by_location.get(cluster_points[node_idx], {}).get('orders', []),
                            cluster_demands[node_idx],
                            cluster_time_windows[node_idx],
                            "Location dropped due to constraints",
                            vehicle.vehiclecode
                        )
                        
                # Extract the route from the solution
                index = cluster_routing.Start(0)
                while not cluster_routing.IsEnd(index):
                    node_index = cluster_manager.IndexToNode(index)
                    route.append(cluster_points[node_index])

                    if node_index > 0 and node_index != vehicle.start_index and node_index != vehicle.end_index:
                        route_load += cluster_demands[node_index]

                    next_index = solution.Value(cluster_routing.NextVar(index))
                    if not cluster_routing.IsEnd(next_index):
                        next_node = cluster_manager.IndexToNode(next_index)
                        route_distance += cluster_distance_matrix[node_index][next_node]
                        time_matrix_list.append(cluster_time_matrix[node_index][next_node])

                    time_var = time_dimension.CumulVar(index)
                    time_min = solution.Min(time_var)
                    time_max = solution.Max(time_var)
                    time_list.append((time_min, time_max))

                    index = next_index
                    
                if diff_start:
                    if distance_start_end > 15:
                        route.insert(1,cluster_points[vehicle.end_index])
                    
                # Add final end location
                route.append(cluster_points[vehicle.end_index])
                
                # Store route data
                all_routes.append(route)
                all_route_vehicle.append(vehicle)
                
                # print(f"route : {len(route)}")
                # print(f"route : {route}")
                # for point_idx, point in enumerate(route[1:], 1):  # Skip first point (depot)
                #     # Ensure the point is valid and has orders
                #     if point in orders_by_location:
                #         orders_data = orders_by_location[point]
                #         # If this location is not in cluster_points, add to unallocated
                #         if point not in cluster_points:
                #             print(point)
                #             add_to_unallocated(
                #                 location=point,
                #                 demand=orders_data['total_volume'],
                #                 orders=orders_data['orders'],
                #                 time_window=orders_data['time_window'],
                #                 reason=f'Location not assigned to cluster',
                #                 vehicle_code=vehicle.vehiclecode
                #             )
            else:
                # If no solution found for this cluster, mark locations as unallocated
                for point_idx, point in enumerate(cluster_points[1:], 1):  # Skip depot
                    if point in orders_by_location:
                        location_data = orders_by_location[point]

                        add_to_unallocated(
                            location= point,
                            demand= cluster_demands[point_idx],
                            orders= location_data['orders'],
                            time_window= cluster_time_windows[point_idx],
                            reason= 'No feasible route found within constraints'
                        )

    # Check for locations that are not in unallocated and not in any route
    all_routed_locations = set()
    # Collect all locations that were routed
    for route in all_routes:
        all_routed_locations.update(route)

    # Check which locations from orders_by_location are missing
    missing_locations = set()
    for location in orders_by_location.keys():
        if (location not in global_unallocated_locations and 
            location not in all_routed_locations and
            location != depot_location):
            missing_locations.add(location)

    # Add missing locations to unallocated with appropriate reason
    for location in missing_locations:
        if location in orders_by_location:
            location_data = orders_by_location[location]
            add_to_unallocated(
                location=location,
                demand=location_data['total_volume'],
                orders=location_data['orders'],
                time_window=location_data['time_window'],
                reason='Location not assigned to any route'
            )
            if configPlan['debug_output']:
                print(f"Found missing location: {location}")
    # sys.exit()
    setProgress(item=item,progress=40,msg='Calculating routes ..',logs=[])
    if all_routes:
        # Track used vehicles
        used_vehicle_id = set()
        
        total_stops = 0
        total_volume = 0
        total_distance = 0
        
        listUnAllocatedStore = {}

        # Update unallocated summary in map legend
        if unallocated:
            
            for u in unallocated:
                
                if 'orders' in u:
                    for gorder in u['orders']:
                        src = Order.from_dict(gorder)
                        
                        # listGoods = listOrderGoods.
                        listGoods = [orderGoods for orderGoods in listOrderGoods if orderGoods.get('orderId') == src.id]

                        if listGoods == None:
                            continue 

                        volumeBeforeJoin = 0
                        weight = 0
                        volume = 0 
                        qty = 0
                        for lGoods in listGoods:
                            goods = OrderGoods.from_dict(lGoods)
                            weight += (goods.weight/1000) * goods.qty # jadiin gram -> kilogram
                            volumeBeforeJoin += goods.volume * goods.qty
                            volume += goods.volume * goods.qty 
                            qty += goods.qty
                        
                        store = next((Store.from_dict(s) for s in listStore if s['code'] == src.storeCode), Store.from_dict({})) # type: ignore
                        
                        # unallocated order in routes
                        routeOrder = RouteOrder.from_dict({})
                        routeOrder.type = "DROP"
                        routeOrder.id = str(uuid.uuid4())
                        # routeOrder.routeId = routeItem.id
                        routeOrder.orderId = src.id
                        routeOrder.orderCode = src.code
                        routeOrder.orderDate = src.orderDate 
                        routeOrder.storeId = src.storeId
                        routeOrder.storeName = src.storeName
                        routeOrder.storeAddress = src.address
                        routeOrder.storeCode = src.storeCode
                        routeOrder.storeCustomerName = src.storeCustomerName
                        routeOrder.storeSkills = store.skills
                        routeOrder.planId = item.id
                        routeOrder.planCode = item.code
                        routeOrder.taskId = item.taskId
                        routeOrder.taskCode = item.taskCode
                        routeOrder.taskDate = item.taskDate 
                        routeOrder.lat = src.lat
                        routeOrder.lon = src.lon
                        routeOrder.skills = src.skills
                        routeOrder.weight = weight # type: ignore
                        routeOrder.volumeBeforeJoin = volumeBeforeJoin # type: ignore
                        routeOrder.qty = qty # type: ignore
                        # routeOrder.eta = eta.astimezone(tz.gettz('UTC'))
                        # routeOrder.etd = etd.astimezone(tz.gettz('UTC'))
                        # routeOrder.duration = duration
                        # routeOrder.durationInTraffic = duration_in_traffic
                        # routeOrder.distance = distance
                        routeOrder.transaction = src.transDuration
                        routeOrder.idle = src.idleTime
                        # routeOrder.sorting = noStore
                        routeOrder.isFinish = False
                        routeOrder.isFinishWithTrouble = False
                        routeOrder.isAllocated = False
                        routeOrder.isDispatched = False
                        routeOrder.isExecuting = False
                        routeOrder.isFrozen = False
                        routeOrder.isManualChange = False
                        # routeOrder.geometry = geometry
                        # routeOrder.routes = route
                        # routeOrder.durationUsed = duration_used
                        # routeOrder.durationTotal = totalDurationUsed
                        routeOrder.priority = src.priority
                        routeOrder.branchCode = src.branchCode
                        routeOrder.branchColor = src.branchColor
                        routeOrder.branchId = src.branchId
                        routeOrder.branchName = src.branchName
                        routeOrder.elevation = src.elevation
                        routeOrder.kecId = src.kecId
                        routeOrder.kecName = src.kecName
                        routeOrder.kelId = src.kelId
                        routeOrder.kelName = src.kelName
                        routeOrder.kotaId = src.kotaId
                        routeOrder.kotaName = src.kotaName
                        routeOrder.provId = src.provId
                        routeOrder.provName = src.provName
                        routeOrder.note = u.get('reason', 'Unknown')
                        
                        listUnAllocatedStore[src.id] = routeOrder
        
        noTour = 0
        taskVehicleStore = {}
        for i, (route, vehicle) in enumerate(zip(all_routes, all_route_vehicle)):
            
            startDateLocal = vehicle.startDate.astimezone(tz.gettz('Asia/Jakarta')).strftime("%Y-%m-%d")
            endDateLocal = vehicle.endDate.astimezone(tz.gettz('Asia/Jakarta')).strftime("%Y-%m-%d")
            
            vehicle_startDatetime = datetime.strptime(f"{startDateLocal} {vehicle.startTime}","%Y-%m-%d %H:%M")
            vehicle_endDatetime = datetime.strptime(f"{endDateLocal} {vehicle.endTime}","%Y-%m-%d %H:%M")
            
            coordinates = []
            
            for idx, point in enumerate(route):
                lon, lat = map(float, point.split(','))
                coordinates.append([lat, lon])

            route_data = get_route_directions(coordinates, vehicle, table_matrix, vehicle_startDatetime, configPlan['map_option'],item.traffic)
            # print(f"route_data : {route_data}")
            if not route_data:
                continue
            
            
            used_vehicle_id.add(vehicle.id)
            noTour += 1
            
            # Update running totals
            total_stops += len(route) - 1
            # total_volume += load
            # total_distance += distance
            
            route_coords = route_data['routes'][0]['geometry']['coordinates']
            path_coords = [[coord[1], coord[0]] for coord in route_coords]
            
            # Sort coordinates if they don't match route_coords
            # if coordinates != path_coords:
            #     ordered_coords = []
            #     # First add depot
            #     ordered_coords.append(coordinates[0])
                
            #     # Find closest remaining points based on path_coords
            #     remaining_coords = coordinates[1:]
            #     for path_coord in path_coords:
            #         if not remaining_coords:
            #             break
                        
            #         # Find closest point to current path coordinate
            #         min_dist = float('inf')
            #         closest_idx = -1
            #         for i, coord in enumerate(remaining_coords):
            #             dist = ((coord[0] - path_coord[0])**2 + (coord[1] - path_coord[1])**2)**0.5
            #             if dist < min_dist:
            #                 min_dist = dist
            #                 closest_idx = i
                            
            #         if closest_idx >= 0:
            #             ordered_coords.append(remaining_coords.pop(closest_idx))
                        
            #     # Add any remaining points
            #     ordered_coords.extend(remaining_coords)
                
            #     # Update coordinates list
            #     coordinates = ordered_coords
            
            currentVolume = 0
            currentWeight = 0
            currentQty = 0
            totalDistance = 0
            totalDuration = 0
            totalTransaction = 0
            totalDurationInTraffic = 0
            totalDurationUsed = 0
            totalIdle = 0
            noStore = 0
            
            listDataStore = []
            listDataGoods = []
            
            routeItem = Route.from_dict({})
            routeItem.id = str(uuid.uuid4())
            routeItem.code = f"Tour-{noTour}"
            routeItem.driverId = vehicle.driverId
            routeItem.driverName = vehicle.drivername
            routeItem.driverUsername = vehicle.driverusername
            routeItem.driverPhone = vehicle.driverphone
            routeItem.vehicleId = vehicle.vehicleId
            routeItem.vehicleCode = vehicle.vehiclecode
            routeItem.vehiclePlateNumber = vehicle.vehicleplatenumber
            routeItem.transporterId = vehicle.transporterId
            routeItem.transporterCode = vehicle.transportercode
            routeItem.transporterName = vehicle.transportername
            routeItem.transporterPhone = vehicle.transporterphone
            routeItem.vehicleModelId = vehicle.vehicleModelId
            routeItem.vehicleModelCode = vehicle.vehiclemodelcode
            routeItem.planId = item.id
            routeItem.planCode = item.code
            routeItem.planMode = item.mode
            routeItem.planTraffic = item.traffic
            routeItem.isAllocated = True
            routeItem.isDispatched = False
            routeItem.isFrozen = False
            routeItem.route = ""
            routeItem.startDate = vehicle_startDatetime.astimezone(tz.gettz('UTC'))
            routeItem.endDate = vehicle_endDatetime.astimezone(tz.gettz('UTC'))
            routeItem.tourStartDate = vehicle_startDatetime.astimezone(tz.gettz('UTC'))
            routeItem.tourStartDuration = 0
            routeItem.tourStartDistance = 0
            routeItem.tourEndDate = vehicle_startDatetime.astimezone(tz.gettz('UTC'))
            routeItem.tourEndDuration = 0
            routeItem.tourEndDistance = 0
            routeItem.startLocationLat = vehicle.startLocationLat
            routeItem.startLocationLon = vehicle.startLocationLon
            routeItem.startLocationElevation = vehicle.startLocationElevation
            routeItem.startLocationAddress = vehicle.startLocationAddress
            routeItem.endLocationLat = vehicle.endLocationLat
            routeItem.endLocationLon = vehicle.endLocationLon
            routeItem.endLocationElevation = vehicle.endLocationElevation
            routeItem.endLocationAddress = vehicle.endLocationAddress
            routeItem.maxCapacity = vehicle.maxCapacity
            routeItem.maxWeight = vehicle.maxWeight
            routeItem.isFinish = False
            routeItem.isFinishWithTrouble = False
            routeItem.note = ""
            routeItem.branchCode = item.branchCode
            routeItem.branchColor = item.branchColor
            routeItem.branchId = item.branchId
            routeItem.branchName = item.branchName
            routeItem.taskId = item.taskId
            routeItem.taskCode = item.taskCode
            routeItem.taskDate = item.taskDate
            routeItem.travelMode = vehicle.travelMode  
            routeItem.avoidTolls = vehicle.avoidTolls
            routeItem.avoidFerries = vehicle.avoidFerries
            routeItem.avoidHighways = vehicle.avoidHighways
            routeItem.totalDurationAvailable = ((vehicle_startDatetime - vehicle_endDatetime).total_seconds() / 60.0)
            
            start_minutes = convert_time_to_minutes(vehicle.startTime) if vehicle.startTime else 480
            # end_time = convert_time_to_minutes(vehicle.endTime) if vehicle.endTime else 1259
            
            cumulative_time = int(start_minutes) # type: ignore
            
            # Add vehicle start/end locations to cluster points
            start_point = f"{vehicle_locations[vehicle.id]['start'][1]},{vehicle_locations[vehicle.id]['start'][0]}"
            end_point = f"{vehicle_locations[vehicle.id]['end'][1]},{vehicle_locations[vehicle.id]['end'][0]}"
            
            # Check if start point is different from depot
            def haversine_meters(lon1, lat1, lon2, lat2):
                """Calculate distance between two lat/lon points in meters"""
                R = 6371000  # Earth's radius in meters
                lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
                dlon = lon2 - lon1 
                dlat = lat2 - lat1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a))
                return R * c
            
            # Check if start point is different from depot, with tolerance
            start_lon, start_lat = map(float, start_point.split(','))
            end_lon, end_lat = map(float, end_point.split(','))
            depot_lon, depot_lat = map(float, depot_location.split(','))
            distance_start = haversine_meters(start_lon, start_lat, depot_lon, depot_lat)
            distance_end = haversine_meters(end_lon, end_lat, depot_lon, depot_lat)
            
            diff_start = distance_start < 15
            diff_end = distance_end < 15
            
            for idx, point in enumerate(route):
                lon, lat = map(float, point.split(','))
                # --- Print distance from previous stop to this stop ---
                current_distance = 0
                current_duration = 0
                if idx != 0:
                    prev_point = route[idx - 1]
                    prev_idx = idx - 1
                    curr_idx = idx
                    # Try to get distance from route_data if available, else fallback
                    if 'routes' in route_data and 'legs' in route_data['routes'][0]:
                        if prev_idx < len(route_data['routes'][0]['legs']):
                            current_distance = route_data['routes'][0]['legs'][prev_idx]['distance']
                            current_duration = route_data['routes'][0]['legs'][prev_idx]['duration'] // 60
                
                totalDistance += current_distance
                totalDuration += current_duration
                
                currentRoute= ''
                
                # Get current and next stop indices
                current_idx = route.index(point)
                next_idx = current_idx + 1 if current_idx + 1 < len(route) else None
                
                if next_idx is not None:
                    # Get coordinates for current segment
                    if 'routes' in route_data and len(route_data['routes']) > 0:
                        geometry_coordinates = route_data['routes'][0]['geometry']['coordinates']
                        # Find the indices in geometry_coordinates that are closest to the current and next stop

                        def haversine(lon1, lat1, lon2, lat2):
                            # Calculate the great circle distance between two points on the earth (specified in decimal degrees)
                            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
                            dlon = lon2 - lon1
                            dlat = lat2 - lat1
                            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                            c = 2 * asin(sqrt(a))
                            r = 6371000  # Radius of earth in meters
                            return c * r

                        def closest_index(coords, target_lon, target_lat):
                            min_dist = float('inf')
                            min_idx = 0
                            for i, (lon, lat) in enumerate(coords):
                                dist = haversine(lon, lat, target_lon, target_lat)
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = i
                            return min_idx

                        current_lon, current_lat = lon, lat
                        next_lon, next_lat = map(float, route[next_idx].split(','))

                        start_idx = closest_index(geometry_coordinates, current_lon, current_lat)
                        end_idx = closest_index(geometry_coordinates, next_lon, next_lat)

                        if start_idx > end_idx:
                            segment_coordinates = geometry_coordinates[end_idx:start_idx+1]
                        else:
                            segment_coordinates = geometry_coordinates[start_idx:end_idx+1]
                        segment_coordinates = [[lat, lon] for lon, lat in segment_coordinates]
                        currentRoute = json.dumps(segment_coordinates)
                        
                if point == depot_location:  # Depot
                    
                    if idx == len(route) - 1:
                        cumulative_time += current_duration
                        
                        arrival_hours = int(cumulative_time) // 60
                        arrival_minutes = int(cumulative_time) % 60
                        arrival_time_str = f"{arrival_hours:02d}:{arrival_minutes:02d}"
                        eta_local = datetime.strptime(f"{startDateLocal} {arrival_time_str}", "%Y-%m-%d %H:%M")
                        
                        # allocated order in routes
                        # routeOrder = RouteOrder.from_dict({})
                        # routeOrder.type = "LOAD"
                        # routeOrder.id = str(uuid.uuid4())
                        # routeOrder.routeId = routeItem.id
                        # # routeOrder.orderId = ""
                        # # routeOrder.orderCode = ""
                        # # routeOrder.orderDate = ""
                        # # routeOrder.storeId = ""
                        # # routeOrder.storeName = ""
                        # # routeOrder.storeAddress = ""
                        # # routeOrder.storeCode = ""
                        # # routeOrder.storeCustomerName = ""
                        # # routeOrder.storeCustomerPhone = ""
                        # routeOrder.planId = item.id
                        # routeOrder.planCode = item.code
                        # routeOrder.taskId = item.taskId
                        # routeOrder.taskCode = item.taskCode
                        # routeOrder.taskDate = item.taskDate 
                        # routeOrder.lat = item.brancLat
                        # routeOrder.lon = item.brancLon
                        # routeOrder.skills = ""
                        # routeOrder.weight = 0
                        # routeOrder.volume = 0
                        # routeOrder.volumeBeforeJoin = 0
                        # routeOrder.qty = 0
                        # routeOrder.eta = eta_local.astimezone(tz.gettz('UTC'))
                        # routeOrder.etd = eta_local.astimezone(tz.gettz('UTC'))
                        # routeOrder.duration = current_duration
                        # routeOrder.durationInTraffic = current_duration
                        # routeOrder.distance = current_distance
                        # routeOrder.transaction = configPlan['preparation_time'] if idx == 1 else 0
                        # routeOrder.idle = 0
                        # routeOrder.sorting = noStore
                        # routeOrder.realSorting = noStore
                        # routeOrder.isFinish = False
                        # routeOrder.isFinishWithTrouble = False
                        # routeOrder.isAllocated = True
                        # routeOrder.isDispatched = False
                        # routeOrder.isExecuting = False
                        # routeOrder.isFrozen = False
                        # routeOrder.isManualChange = False
                        # routeOrder.routes = currentRoute
                        # routeOrder.durationUsed = configPlan['preparation_time'] if idx == 1 else 0
                        # routeOrder.durationTotal = configPlan['preparation_time'] if idx == 1 else 0
                        # routeOrder.priority = 0
                        # routeOrder.branchCode = item.branchCode
                        # routeOrder.branchColor = item.branchColor
                        # routeOrder.branchId = src.branchId
                        # routeOrder.branchName = item.branchName
                        # routeOrder.elevation = 0
                        # routeOrder.kecId = ""
                        # routeOrder.kecName = ""
                        # routeOrder.kelId = ""
                        # routeOrder.kelName = ""
                        # routeOrder.kotaId = ""
                        # routeOrder.kotaName = ""
                        # routeOrder.provId = ""
                        # routeOrder.provName = ""
                        # listDataStore.append(routeOrder)
                    else:
                        # Get current and next stop indices
                        arrival_hours = int(cumulative_time) // 60
                        arrival_minutes = int(cumulative_time) % 60
                        arrival_time_str = f"{arrival_hours:02d}:{arrival_minutes:02d}"
                        eta_local = datetime.strptime(f"{startDateLocal} {arrival_time_str}", "%Y-%m-%d %H:%M")
                        
                        preparation_time = configPlan['preparation_time'] if idx == 1 else 0
                        cumulative_time += preparation_time
                        totalTransaction += preparation_time
                        
                        departure_hours = int(cumulative_time) // 60
                        departure_minutes = int(cumulative_time) % 60
                        departure_time_str = f"{departure_hours:02d}:{departure_minutes:02d}"
                        etd_local = datetime.strptime(f"{startDateLocal} {departure_time_str}", "%Y-%m-%d %H:%M")
                        
                        # allocated order in routes
                        # routeOrder = RouteOrder.from_dict({})
                        # routeOrder.type = "LOAD"
                        # routeOrder.id = str(uuid.uuid4())
                        # routeOrder.routeId = routeItem.id
                        # routeOrder.routeCode = routeItem.code
                        # # routeOrder.orderId = ""
                        # # routeOrder.orderCode = ""
                        # # routeOrder.orderDate = ""
                        # # routeOrder.storeId = ""
                        # # routeOrder.storeName = ""
                        # # routeOrder.storeAddress = ""
                        # # routeOrder.storeCode = ""
                        # # routeOrder.storeCustomerName = ""
                        # # routeOrder.storeCustomerPhone = ""
                        # routeOrder.planId = item.id
                        # routeOrder.planCode = item.code
                        # routeOrder.taskId = item.taskId
                        # routeOrder.taskCode = item.taskCode
                        # routeOrder.taskDate = item.taskDate 
                        # routeOrder.lat = item.brancLat
                        # routeOrder.lon = item.brancLon
                        # routeOrder.skills = ""
                        # routeOrder.weight = 0
                        # routeOrder.volume = 0
                        # routeOrder.volumeBeforeJoin = 0
                        # routeOrder.qty = 0
                        # routeOrder.eta = eta_local.astimezone(tz.gettz('UTC'))
                        # routeOrder.etd = etd_local.astimezone(tz.gettz('UTC'))
                        # routeOrder.duration = current_duration
                        # routeOrder.durationInTraffic = current_duration
                        # routeOrder.distance = current_distance
                        # routeOrder.transaction = configPlan['preparation_time'] if idx == 1 else 0
                        # routeOrder.idle = 0
                        # routeOrder.sorting = noStore
                        # routeOrder.realSorting = noStore
                        # routeOrder.isFinish = False
                        # routeOrder.isFinishWithTrouble = False
                        # routeOrder.isAllocated = True
                        # routeOrder.isDispatched = False
                        # routeOrder.isExecuting = False
                        # routeOrder.isFrozen = False
                        # routeOrder.isManualChange = False
                        # routeOrder.routes = currentRoute
                        # routeOrder.durationUsed = configPlan['preparation_time'] if idx == 1 else 0
                        # routeOrder.durationTotal = configPlan['preparation_time'] if idx == 1 else 0
                        # routeOrder.priority = 0
                        # routeOrder.branchCode = item.branchCode
                        # routeOrder.branchColor = item.branchColor
                        # routeOrder.branchId = src.branchId
                        # routeOrder.branchName = item.branchName
                        # routeOrder.elevation = 0
                        # routeOrder.kecId = ""
                        # routeOrder.kecName = ""
                        # routeOrder.kelId = ""
                        # routeOrder.kelName = ""
                        # routeOrder.kotaId = ""
                        # routeOrder.kotaName = ""
                        # routeOrder.provId = ""
                        # routeOrder.provName = ""
                        # listDataStore.append(routeOrder)
                # elif point == start_point:
                #     arrival_hours = int(cumulative_time) // 60
                #     arrival_minutes = int(cumulative_time) % 60
                #     arrival_time_str = f"{arrival_hours:02d}:{arrival_minutes:02d}"
                #     eta_local = datetime.strptime(f"{startDateLocal} {arrival_time_str}", "%Y-%m-%d %H:%M")
                    
                #     # allocated order in routes
                #     # routeOrder = RouteOrder.from_dict({})
                #     # routeOrder.type = "START"
                #     # routeOrder.id = str(uuid.uuid4())
                #     # routeOrder.routeId = routeItem.id
                #     # routeOrder.routeCode = routeItem.code
                #     # # routeOrder.orderId = ""
                #     # # routeOrder.orderCode = ""
                #     # # routeOrder.orderDate = ""
                #     # # routeOrder.storeId = ""
                #     # # routeOrder.storeName = ""
                #     # # routeOrder.storeAddress = ""
                #     # # routeOrder.storeCode = ""
                #     # # routeOrder.storeCustomerName = ""
                #     # # routeOrder.storeCustomerPhone = ""
                #     # routeOrder.planId = item.id
                #     # routeOrder.planCode = item.code
                #     # routeOrder.taskId = item.taskId
                #     # routeOrder.taskCode = item.taskCode
                #     # routeOrder.taskDate = item.taskDate 
                #     # routeOrder.lat = item.brancLat
                #     # routeOrder.lon = item.brancLon
                #     # routeOrder.skills = ""
                #     # routeOrder.weight = 0
                #     # routeOrder.volume = 0
                #     # routeOrder.volumeBeforeJoin = 0
                #     # routeOrder.qty = 0
                #     # routeOrder.eta = eta_local.astimezone(tz.gettz('UTC'))
                #     # routeOrder.etd = eta_local.astimezone(tz.gettz('UTC'))
                #     # routeOrder.duration = current_duration
                #     # routeOrder.durationInTraffic = current_duration
                #     # routeOrder.distance = current_distance
                #     # routeOrder.transaction = 0
                #     # routeOrder.idle = 0
                #     # routeOrder.sorting = noStore
                #     # routeOrder.realSorting = noStore
                #     # routeOrder.isFinish = False
                #     # routeOrder.isFinishWithTrouble = False
                #     # routeOrder.isAllocated = True
                #     # routeOrder.isDispatched = False
                #     # routeOrder.isExecuting = False
                #     # routeOrder.isFrozen = False
                #     # routeOrder.isManualChange = False
                #     # routeOrder.routes = currentRoute
                #     # routeOrder.durationUsed = 0
                #     # routeOrder.durationTotal = 0
                #     # routeOrder.priority = 0
                #     # routeOrder.branchCode = item.branchCode
                #     # routeOrder.branchColor = item.branchColor
                #     # routeOrder.branchId = src.branchId
                #     # routeOrder.branchName = item.branchName
                #     # routeOrder.elevation = 0
                #     # routeOrder.kecId = ""
                #     # routeOrder.kecName = ""
                #     # routeOrder.kelId = ""
                #     # routeOrder.kelName = ""
                #     # routeOrder.kotaId = ""
                #     # routeOrder.kotaName = ""
                #     # routeOrder.provId = ""
                #     # routeOrder.provName = ""
                #     # listDataStore.append(routeOrder)
                    
                else:  # Stop in cluster
                    stop_number = idx  # Stop number in route
                    route_orders = orders_by_location[point]['orders']
                    arrival_time = cumulative_time + current_duration
                    last_departure_time_order = 0
                    
                    for gorder in route_orders:
                        order = Order.from_dict(gorder)
                        
                        # listGoods = listOrderGoods.
                        listGoods = [orderGoods for orderGoods in listOrderGoods if orderGoods.get('orderId') == order.id]

                        if listGoods == None:
                            continue 
                        
                        # Join Capacity Items
                        joinCapacity = canJoinCapacity(listDataGoods,listGoods)

                        volumeBeforeJoin = 0
                        weight = 0
                        volume = 0 
                        qty = 0
                        for lGoods in listGoods:
                            goods = OrderGoods.from_dict(lGoods)
                            weight += (goods.weight/1000) * goods.qty # jadiin gram -> kilogram
                            volumeBeforeJoin += goods.volume * goods.qty
                            if len(joinCapacity) > 0:
                                jcId = []
                                for jc in joinCapacity:
                                    jcId.append(jc['id']) 
                                if goods.id in jcId: 
                                    volume += 0
                                else:
                                    volume += goods.volume * goods.qty 
                            else:
                                volume += goods.volume * goods.qty 
                                
                            qty += goods.qty
                            listDataGoods.append(lGoods)
                            
                        
                        currentWeight += weight
                        currentVolume += volume 
                        currentQty += qty
                        totalIdle += order.idleTime if order.idleTime is not None else 0
                        
                        # Calculate ETA (arrival time) and ETD (departure time) in UTC
                        if gorder == route_orders[0]:
                            # Convert arrival_time (in minutes) to HH:MM format
                            arrival_hours = int(arrival_time) // 60
                            arrival_minutes = int(arrival_time) % 60
                            arrival_time_str = f"{arrival_hours:02d}:{arrival_minutes:02d}"
                            # Handle arrival hours > 24 by incrementing the date
                            days_to_add = arrival_hours // 24
                            arrival_hours = arrival_hours % 24
                            arrival_time_str = f"{arrival_hours:02d}:{arrival_minutes:02d}"
                            eta_local = datetime.strptime(f"{startDateLocal} {arrival_time_str}", "%Y-%m-%d %H:%M")
                            if days_to_add > 0:
                                eta_local += timedelta(days=days_to_add)
                        else:
                            departure_hours = int(last_departure_time_order) // 60
                            departure_minutes = int(last_departure_time_order) % 60
                            departure_time_str = f"{departure_hours:02d}:{departure_minutes:02d}"
                            # Handle departure hours > 24 by incrementing the date
                            days_to_add = departure_hours // 24
                            departure_hours = departure_hours % 24
                            departure_time_str = f"{departure_hours:02d}:{departure_minutes:02d}" 
                            eta_local = datetime.strptime(f"{startDateLocal} {departure_time_str}", "%Y-%m-%d %H:%M")
                            if days_to_add > 0:
                                eta_local += timedelta(days=days_to_add)
                        
                        if gorder == route_orders[0]:
                            last_departure_time_order = arrival_time + max(Order.from_dict(order).transDuration for order in route_orders)
                            totalTransaction += max(Order.from_dict(order).transDuration for order in route_orders)
                        else:
                            last_departure_time_order += 0.5
                            totalTransaction += 0.5

                        departure_hours = int(last_departure_time_order) // 60
                        departure_minutes = int(last_departure_time_order) % 60
                        departure_time_str = f"{departure_hours:02d}:{departure_minutes:02d}"
                        # Handle departure hours > 24 by incrementing the date
                        days_to_add = departure_hours // 24
                        departure_hours = departure_hours % 24
                        departure_time_str = f"{departure_hours:02d}:{departure_minutes:02d}" 
                        etd_local = datetime.strptime(f"{startDateLocal} {departure_time_str}", "%Y-%m-%d %H:%M")
                        if days_to_add > 0:
                            etd_local += timedelta(days=days_to_add)
                        
                        # Extract partial geometry for this stop's segment
                        if gorder != route_orders[0]:  # Only for first order at each stop
                            currentRoute = json.dumps([[lon, lat],[lon, lat]])
                        
                        # Get store details for this order's store code
                        store = next((Store.from_dict(s) for s in listStore if s['code'] == order.storeCode), Store.from_dict({})) # type: ignore
                        
                        # allocated order in routes
                        routeOrder = RouteOrder.from_dict({})
                        routeOrder.type = "DROP"
                        routeOrder.id = str(uuid.uuid4())
                        routeOrder.routeId = routeItem.id
                        routeOrder.routeCode = routeItem.code
                        routeOrder.orderId = order.id
                        routeOrder.orderCode = order.code
                        routeOrder.orderDate = order.orderDate 
                        routeOrder.storeId = order.storeId
                        routeOrder.storeName = order.storeName
                        routeOrder.storeAddress = order.address
                        routeOrder.storeCode = order.storeCode
                        routeOrder.storeCustomerName = order.storeCustomerName
                        routeOrder.storeSkills = store.skills
                        routeOrder.planId = item.id
                        routeOrder.planCode = item.code
                        routeOrder.taskId = item.taskId
                        routeOrder.taskCode = item.taskCode
                        routeOrder.taskDate = item.taskDate 
                        routeOrder.lat = order.lat
                        routeOrder.lon = order.lon
                        routeOrder.skills = order.skills
                        routeOrder.weight = weight
                        routeOrder.volume = volume
                        routeOrder.volumeBeforeJoin = volumeBeforeJoin
                        routeOrder.qty = qty
                        routeOrder.eta = eta_local.astimezone(tz.gettz('UTC'))
                        routeOrder.etd = etd_local.astimezone(tz.gettz('UTC'))
                        routeOrder.duration = current_duration * 60 if gorder == route_orders[0] else 0
                        routeOrder.durationInTraffic = current_duration * 60 if gorder == route_orders[0] else 0
                        routeOrder.distance = current_distance if gorder == route_orders[0] else 0
                        routeOrder.transaction = order.transDuration if gorder == route_orders[0] else 0.5 
                        routeOrder.idle = order.idleTime
                        routeOrder.sorting = noStore
                        routeOrder.realSorting = noStore
                        routeOrder.isFinish = False
                        routeOrder.isFinishWithTrouble = False
                        routeOrder.isAllocated = True
                        routeOrder.isDispatched = False
                        routeOrder.isExecuting = False
                        routeOrder.isFrozen = False
                        routeOrder.isManualChange = False
                        routeOrder.routes = currentRoute
                        routeOrder.durationUsed = last_departure_time_order + 0.5
                        routeOrder.durationTotal = last_departure_time_order + 0.5
                        routeOrder.priority = order.priority
                        routeOrder.branchCode = order.branchCode
                        routeOrder.branchColor = order.branchColor
                        routeOrder.branchId = order.branchId
                        routeOrder.branchName = order.branchName
                        routeOrder.elevation = order.elevation
                        routeOrder.kecId = order.kecId
                        routeOrder.kecName = order.kecName
                        routeOrder.kelId = order.kelId
                        routeOrder.kelName = order.kelName
                        routeOrder.kotaId = order.kotaId
                        routeOrder.kotaName = order.kotaName
                        routeOrder.provId = order.provId
                        routeOrder.provName = order.provName 
                        listDataStore.append(routeOrder)
                        for lGoods in listGoods: 
                            listDataGoods.append(lGoods)
                        
                        noStore += 1
                    
                    cumulative_time = last_departure_time_order
                
            cumulative_time_hours = int(cumulative_time) // 60
            cumulative_time_minutes = int(cumulative_time) % 60
            cumulative_time_str = f"{cumulative_time_hours:02d}:{cumulative_time_minutes:02d}"
            # Handle hours > 24 by incrementing the date accordingly
            days_to_add = cumulative_time_hours // 24
            adjusted_hours = cumulative_time_hours % 24
            adjusted_time_str = f"{adjusted_hours:02d}:{cumulative_time_minutes:02d}"
            
            eta_local = datetime.strptime(f"{startDateLocal} {adjusted_time_str}", "%Y-%m-%d %H:%M")
            if days_to_add > 0:
                eta_local += timedelta(days=days_to_add)
            routeItem.isAllocated = True 
            routeItem.tourEndDate = eta_local.astimezone(tz.gettz('UTC'))
            # print(f"routeItem.tourEndDate : {routeItem.tourEndDate.astimezone(tz.gettz('UTC'))}")
            # print(f"routeItem.tourEndDate (local): {routeItem.tourEndDate.astimezone(tz.gettz('Asia/Jakarta'))}")
            # routeItem.tourEndDate = 
            # Calculate route end duration as total travel time in minutes
            routeItem.tourEndDuration = cumulative_time
            routeItem.tourEndDistance = current_distance # type: ignore

            routeItem.totalDistance = totalDistance
            routeItem.totalDuration = totalDuration * 60
            routeItem.totalIdle = totalIdle
            routeItem.totalOrder = len(listDataStore)
            # Calculate total service time for this route
            routeItem.totalTransaction = totalTransaction
            
            routeItem.totalVolume = currentVolume
            routeItem.totalWeight = currentWeight
            routeItem.totalQty = currentQty
            # Reverse each coordinate from [lon, lat] to [lat, lon] before saving
            routeItem.route = json.dumps([[lat, lon] for lon, lat in route_data['routes'][0]['geometry']['coordinates']])
            # routeItem.geometry = route_data['routes'][0]['geometry']
            
            taskVehicleStore[vehicle.id] = {
                "tour": routeItem,
                "stores": listDataStore
            }

        if len(used_vehicle_id) < len(listGroupVehicle):
            unused_vehicles = [gv for gv in listGroupVehicle if gv['id'] not in used_vehicle_id]
            # After printing all route summaries, show unused vehicles
            # print_unused_vehicles(unused_vehicles, vehicle_capacities)
            
            noTour = len(used_vehicle_id)

            for idx, gVehicle in enumerate(unused_vehicles):
                vehicle = GroupVehicle.from_dict(gVehicle)
                noTour += 1
                startDateLocal = vehicle.startDate.astimezone(tz.gettz('Asia/Jakarta')).strftime("%Y-%m-%d")
                endDateLocal = vehicle.endDate.astimezone(tz.gettz('Asia/Jakarta')).strftime("%Y-%m-%d")
                
                vehicle_startDatetime = datetime.strptime(f"{startDateLocal} {vehicle.startTime}","%Y-%m-%d %H:%M")
                vehicle_endDatetime = datetime.strptime(f"{endDateLocal} {vehicle.endTime}","%Y-%m-%d %H:%M")
                
                routeItem = Route.from_dict({})
                routeItem.id = str(uuid.uuid4())
                routeItem.code = f"Tour-{noTour}"
                routeItem.driverId = vehicle.driverId
                routeItem.driverName = vehicle.drivername
                routeItem.driverUsername = vehicle.driverusername
                routeItem.driverPhone = vehicle.driverphone
                routeItem.vehicleId = vehicle.vehicleId
                routeItem.vehicleCode = vehicle.vehiclecode
                routeItem.vehiclePlateNumber = vehicle.vehicleplatenumber
                routeItem.transporterId = vehicle.transporterId
                routeItem.transporterCode = vehicle.transportercode
                routeItem.transporterName = vehicle.transportername
                routeItem.transporterPhone = vehicle.transporterphone
                routeItem.vehicleModelId = vehicle.vehicleModelId
                routeItem.vehicleModelCode = vehicle.vehiclemodelcode
                routeItem.planId = item.id
                routeItem.planCode = item.code
                routeItem.planMode = item.mode
                routeItem.planTraffic = item.traffic
                routeItem.isAllocated = True
                routeItem.isDispatched = False
                routeItem.isFrozen = False
                routeItem.route = ""
                routeItem.startDate = vehicle_startDatetime.astimezone(tz.gettz('UTC'))
                routeItem.endDate = vehicle_endDatetime.astimezone(tz.gettz('UTC'))
                routeItem.tourStartDate = vehicle_startDatetime.astimezone(tz.gettz('UTC'))
                routeItem.tourStartDuration = 0
                routeItem.tourStartDistance = 0
                routeItem.tourEndDate = vehicle_startDatetime.astimezone(tz.gettz('UTC'))
                routeItem.tourEndDuration = 0
                routeItem.tourEndDistance = 0
                routeItem.startLocationLat = vehicle.startLocationLat
                routeItem.startLocationLon = vehicle.startLocationLon
                routeItem.startLocationElevation = vehicle.startLocationElevation
                routeItem.startLocationAddress = vehicle.startLocationAddress
                routeItem.endLocationLat = vehicle.endLocationLat
                routeItem.endLocationLon = vehicle.endLocationLon
                routeItem.endLocationElevation = vehicle.endLocationElevation
                routeItem.endLocationAddress = vehicle.endLocationAddress
                routeItem.maxCapacity = vehicle.maxCapacity
                routeItem.maxWeight = vehicle.maxWeight
                routeItem.isFinish = False
                routeItem.isFinishWithTrouble = False
                routeItem.note = ""
                routeItem.branchCode = item.branchCode
                routeItem.branchColor = item.branchColor
                routeItem.branchId = item.branchId
                routeItem.branchName = item.branchName
                routeItem.taskId = item.taskId
                routeItem.taskCode = item.taskCode
                routeItem.taskDate = item.taskDate
                routeItem.travelMode = vehicle.travelMode  
                routeItem.avoidTolls = vehicle.avoidTolls
                routeItem.avoidFerries = vehicle.avoidFerries
                routeItem.avoidHighways = vehicle.avoidHighways
                routeItem.totalDurationAvailable = ((vehicle_startDatetime - vehicle_endDatetime).total_seconds() / 60.0)
                routeItem.note = "No Tasks Allocated"
                routeItem.isAllocated = False 
                
                taskVehicleStore[vehicle.id] = {
                    "tour": routeItem,
                    "stores": []
                }

        if configPlan['debug_output'] is True:
            print_vehicle_utilization_report(taskVehicleStore, {gv['id']: gv for gv in listGroupVehicle})
            
            generate_route_map(item, taskVehicleStore, depot_location, listStore, listUnAllocatedStore)
            
            # for veId in taskVehicleStore: 
            #     src = taskVehicleStore[veId]
            #     tour = src["tour"]
            #     # Sum duration on all stores in this tour
            #     total_store_duration = sum(getattr(store, 'duration', 0) for store in src["stores"])
            #     total_store_durationInTraffic = sum(getattr(store, 'durationInTraffic', 0) for store in src["stores"])
            #     # print(f"Total duration on stores for tour {tour.code}: {total_store_duration}")
            #     print(f"tour totalDuration: {getattr(tour, 'totalDuration', None)} min ({getattr(tour, 'totalDuration', 0) // 60}h {getattr(tour, 'totalDuration', 0) % 60}m)")
            #     print(f"store duration: {total_store_duration}")
            #     print(f"store durationInTraffic: {total_store_durationInTraffic}")
            
            # Save taskVehicleStore to JSON for debugging/analysis
            debug_json_path = os.path.join("./reports", f"taskVehicleStore_{item.taskId}_{item.id}.json")
            with open(debug_json_path, "w", encoding="utf-8") as f:
                # Convert objects to dict for JSON serialization
                def serialize(obj):
                    if hasattr(obj, "to_dict"):
                        return obj.to_dict()
                    elif hasattr(obj, "__dict__"):
                        return obj.__dict__
                    else:
                        return str(obj)
                json.dump(
                    {k: {"tour": serialize(v["tour"]), "stores": [serialize(s) for s in v["stores"]]} for k, v in taskVehicleStore.items()},
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=serialize
                )
            print(f"Saved taskVehicleStore to {debug_json_path}")
        else:
            setProgress(item=item,progress=98,msg='Saving Route',logs=[])
            '''
            Save Route && Order Store
            ''' 
            totalAllocated = 0
            totalDistance = 0
            totalDuration = 0
            totalDurationAvailable = 0
            totalIdle = 0
            totalQty = 0
            totalStore = 0
            totalTransaction = 0
            totalVehicle = 0
            totalVehicleUsed = 0
            totalVolume = 0
            totalWeight = 0
            for veId in taskVehicleStore: 
                src = taskVehicleStore[veId]
                tour = src["tour"]
                print(f"{tour.code} | Store Count : {len(src['stores'])}")
                insertRoute(item=tour)
                stores = src["stores"]
                totalAllocated += len(stores)
                if tour.totalDistance:
                    totalDistance += tour.totalDistance
                if tour.totalDistance:
                    totalDuration += tour.totalDuration
                if tour.totalDurationAvailable:
                    totalDurationAvailable += tour.totalDurationAvailable
                if tour.totalIdle:
                    totalIdle += tour.totalIdle
                if tour.totalQty:
                    totalQty += tour.totalQty
                if tour.totalTransaction:
                    totalTransaction += tour.totalTransaction
                if tour.totalVolume:
                    totalVolume += tour.totalVolume
                if tour.totalWeight:
                    totalWeight += tour.totalWeight
                totalStore += len(stores)
                totalVehicle += 1
                if len(stores) > 0:
                    totalVehicleUsed += 1

                for store in stores:
                    # print(f">> Save Store {store.id}")
                    insertRouteOrder(item=store)

            setProgress(item=item,progress=99,msg='Saving Task',logs=[])
            totalStore += len(listUnAllocatedStore)
            '''
            Save UnAllocated Order Store
            ''' 
            print(f"Unallocated Store Count : {len(listUnAllocatedStore)}")
            for storeId in listUnAllocatedStore:
                store = listUnAllocatedStore[storeId]
                insertRouteOrder(item=store)
            
            '''
            Update Plan
            ''' 
            plan = Plan.from_dict({})
            plan.id = item.id
            plan.totalAllocated = totalAllocated
            plan.totalUnallocated = len(listUnAllocatedStore)
            plan.totalDistance = totalDistance
            plan.totalDuration = totalDuration
            plan.totalDurationAvailable = totalDurationAvailable
            plan.totalIdle = totalIdle
            plan.totalQty = totalQty
            plan.totalStore = totalStore
            plan.totalTransaction = totalTransaction
            plan.totalVehicle = totalVehicle
            plan.totalVehicleUsed = totalVehicleUsed
            plan.totalVolume = totalVolume
            plan.totalWeight = totalWeight
            # plan.cluster = json.dumps(planCluster)
            savePlan(item=plan)
        return all_routes
    else:
        setError(item, ['No solution found!'])
        return None
    
# Add helper function for vehicle utilization reporting
def print_vehicle_utilization_report(taskVehicleStore, vehicle_data=None):
    """Generate a detailed vehicle utilization report"""
    report = []
    total_volume = 0
    total_distance = 0
    total_duration = 0
    total_stores = 0
    total_vehicles = len(taskVehicleStore)
    active_vehicles = 0
    
    report.append("\nVehicle Utilization Report")
    report.append("=" * 80)
    
    for vehId, data in taskVehicleStore.items():
        tour = data["tour"]
        stores = data["stores"]
        
        # Skip empty tours
        if not stores:
            continue
            
        active_vehicles += 1
        total_stores += len(stores)
        
        # Calculate utilization metrics
        volume_util = (tour.totalVolume or 0) / (tour.maxCapacity * 100000) * 100 if tour.maxCapacity else 0
        time_util = ((tour.totalDuration or 0) / 60 ) / ((tour.endDate - tour.startDate).total_seconds() / 60) * 100
        
        # Accumulate totals
        total_volume += tour.totalVolume or 0
        total_distance += tour.totalDistance or 0
        total_duration += (tour.totalDuration or 0) / 60
        
        report.append(f"\nVehicle: {tour.vehicleModelCode} ({tour.code})")
        report.append("-" * 40)
        report.append(f"Stops: {len(stores)}")
        report.append(f"Volume: {tour.totalVolume/100000:.2f}m³ ({volume_util:.1f}% capacity)")
        report.append(f"Distance: {tour.totalDistance/1000:.2f}km")
        report.append(f"Duration: {((tour.totalDuration or 0) / 60 )//60}h {((tour.totalDuration or 0) / 60 )%60}m ({time_util:.1f}% time)")
        
        if vehicle_data and vehId in vehicle_data:
            veh = vehicle_data[vehId]
            report.append(f"Working Hours: {veh.get('startTime', 'N/A')} - {veh.get('endTime', 'N/A')}")
    
    # Add summary
    report.append("\nSummary")
    report.append("=" * 40)
    report.append(f"Total Vehicles: {total_vehicles}")
    report.append(f"Active Vehicles: {active_vehicles}")
    report.append(f"Total Stops: {total_stores}")
    report.append(f"Total Volume: {total_volume/100000:.2f}m³")
    report.append(f"Total Distance: {total_distance/1000:.2f}km")
    report.append(f"Total Duration: {total_duration//60}h {total_duration%60}m")
    
    print("\n".join(report))

def generate_route_map(item:Plan, taskVehicleStore, depot_location, listStore, unallocated=None):
    """Generate an enhanced route visualization map"""
    depot_lat, depot_lon = map(float, depot_location.split(',')[::-1])
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=12)
    
    # Define color scheme
    colors = ['#FF4B4B', '#4B4BFF', '#4BFF4B', '#FF4BFF', '#FFB84B', '#4BFFFF']
    
    # Add depot marker
    folium.Marker(
        [depot_lat, depot_lon],
        popup='Depot',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # Plot routes
    for vehId, data in taskVehicleStore.items():
        tour = data["tour"]
        stores = data["stores"]
        
        if not stores:
            continue
            
        # Assign unique color based on vehicle index to ensure different colors
        color = colors[list(taskVehicleStore.keys()).index(vehId) % len(colors)]
        
        # Create route line if route geometry exists
        if tour.route:
            try:
                route_coords = json.loads(tour.route)
                # Draw route line
                folium.PolyLine(
                    route_coords,
                    weight=3,
                    color=color,
                    opacity=0.8,
                    popup=f"""
                        <b>{tour.code}</b><br>
                        Vehicle: {tour.vehicleModelCode}<br>
                        Volume: {tour.totalVolume/100000:.2f}m³<br>
                        Distance: {tour.totalDistance/1000:.2f}km<br>
                        Duration: {tour.totalDuration//60}h {tour.totalDuration%60}m
                    """
                ).add_to(m)
                
                # Add arrow markers along the route
                plugins.AntPath(
                    route_coords,
                    color=color,
                    weight=2,
                    opacity=0.6,
                    delay=1000
                ).add_to(m)
            except Exception as e:
                print(f"Warning: Could not plot route for {tour.code}: {str(e)}")
        
        # Add store markers
        # Group stores by location
        location_stores = {}
        for store in stores:
            if store.type == "DROP":
                # print(f"store.duration : {store.duration}")
                loc_key = f"{store.lat},{store.lon}"
                if loc_key not in location_stores:
                    location_stores[loc_key] = []
                location_stores[loc_key].append(store)

        # Add markers for each location
        for idx, (loc_key, loc_stores) in enumerate(location_stores.items(), 1):
            lat, lon = map(float, loc_key.split(','))
            
            # Calculate aggregated metrics for this location
            total_volume = sum(s.volume for s in loc_stores)
            earliest_eta = min(s.eta for s in loc_stores)
            earliest_etd = max(s.etd for s in loc_stores)
            total_duration = sum(s.duration for s in loc_stores) if loc_stores[0].duration else 0
            store_codes = ", ".join(set(s.storeCode for s in loc_stores))
            
            # Check if store is open with enhanced validation
            # store_time = getStoreOpenTime(listStore=listStore, order=order)
            
                       
            # storeTime = (60, 1439)  # Store is not configure operational time (01:00 to 23:59)
            open_time = ""
            close_time = ""
            store = next((Store.from_dict(s) for s in listStore if s['code'] in set(s.storeCode for s in loc_stores)))
            if store.dayStatus == "1":
                now = earliest_eta.strftime("%A").lower()
                open_time = getattr(store, f"{now}Start", None)
                close_time = getattr(store, f"{now}End", None)
                # open_minutes = convert_time_to_minutes(open_time)
                # close_minutes = convert_time_to_minutes(close_time)
                # storeTime = 
            
            folium.CircleMarker(
            [lat, lon],
            radius=8,
            color=color,
            fill=True,
            popup=f"""
                <b>{tour.code} - Stop {idx}</b><br>
                Store(s): {store_codes}<br>
                open: {open_time}<br>
                close: {close_time}<br>
                Orders: {len(loc_stores)}<br>
                Volume: {total_volume:.2f}m³<br>
                ETA: {earliest_eta.astimezone(tz.gettz('Asia/Jakarta')).strftime('%Y-%m-%d %H:%M')}<br>
                ETD: {earliest_etd.astimezone(tz.gettz('Asia/Jakarta')).strftime('%Y-%m-%d %H:%M')}<br>
                Duration: {total_duration}min
            """
            ).add_to(m)
    
    # Add unallocated stores if provided
    if unallocated and isinstance(unallocated, dict):
        for store_id, store in unallocated.items():
            try:
                lat = store.lat 
                lon = store.lon
                volume = store.volume/100000 if store.volume else 0
                
                folium.Marker(
                    [lat, lon],
                    popup=f"""
                        <b>UNALLOCATED</b><br>
                        Store: {store.storeCode}<br>
                        Volume: {volume:.2f}m³<br>
                        Note: {store.note}
                    """,
                    icon=folium.Icon(color='black', icon='warning-sign')
                ).add_to(m)
            except Exception as e:
                print(f"Warning: Could not plot unallocated store: {str(e)}")
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; 
                background-color: white; padding: 10px; border: 2px solid grey; 
                border-radius: 5px;">
        <p><strong>Route Legend</strong></p>
    '''
    
    for vehId, data in taskVehicleStore.items():
        tour = data["tour"]
        if not data["stores"]:
            continue
        color = colors[list(taskVehicleStore.keys()).index(vehId) % len(colors)]
        legend_html += f'''
            <p>
                <span style="color: {color};">■</span>
                {tour.code} - {tour.vehicleModelCode}<br/>
                <span style="font-size: 0.8em; margin-left: 15px;">
                    {len(data["stores"])} stops, 
                    {tour.totalVolume/100000:.1f}m³
                </span>
            </p>
        '''
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html)) # type: ignore
    
    # Save map
    os.makedirs("./reports", exist_ok=True)
    map_file = os.path.join("reports", f"route_map_{item.taskId}_{item.id}.html")
    m.save(map_file)

# Join Capacity for Item Category Can Be Join, Ex: Pipa 60cm & Pipa 30cm
def canJoinCapacity(listLoadedGoods,currentGoods):
    itemCanJoin = []
    for llg in listLoadedGoods:
        ag = OrderGoods.from_dict(llg)
        if ag.goodsCategoryJoinCapacity:
            llg['volumeAvailable'] = ag.volume
            llg['joined'] = False
            llg['child'] = []
            llg['isCurrentGoods'] = False
            itemCanJoin.append(llg)
            
    for llg in currentGoods:
        ag = OrderGoods.from_dict(llg)
        if ag.goodsCategoryJoinCapacity:
            llg['volumeAvailable'] = ag.volume
            llg['joined'] = False
            llg['child'] = []
            llg['isCurrentGoods'] = True
            itemCanJoin.append(llg)
    
    newItemCanJoin = sorted(itemCanJoin, key=lambda d: d['volume'])

    # grouped by volume
    groupItemVol = {}
    for llg in newItemCanJoin:
        ag = OrderGoods.from_dict(llg)
        if ag.volume not in groupItemVol:
            groupItemVol[ag.volume] = []
        groupItemVol[ag.volume].append(llg)

    for i1 in groupItemVol:
        src = groupItemVol[i1]
        for src2 in src:
            for i2 in groupItemVol:
                if i1 != i2:
                    src3 = groupItemVol[i2]
                    for src4 in src3:
                        if src2['volume'] <= src4['volumeAvailable'] and src2['joined'] == False:
                            src4['volumeAvailable'] -= src2['volume']
                            src4['child'].append(src2)
                            src2['joined'] = True

    saveVolume = 0
    isCurrentGoodsJoined = False
    currentJoinedOrder = []
    currentJoinTo = []
    newGroup = []
    for i1 in groupItemVol:
        src = groupItemVol[i1]
        newObj = {
            "label": i1,
            "options": []
        }
        for src2 in src:
            if src2['joined'] == False:
                newObj["options"].append(src2)
                currentJoinTo.append(src2)
            else:
                saveVolume += src2['volume']
                if src2['isCurrentGoods']:
                    isCurrentGoodsJoined = True
                    currentJoinedOrder.append(src2)

        newGroup.append(newObj)
    
    return currentJoinedOrder

def setProgress(item: Plan, progress:int,msg:str,logs:list):
    if Config.DEBUG == '1': return
    updateProgress(item.id,Plan.from_dict({
        'progress':progress,
        'progressMsg':msg,
        'progressLog':"\n\n".join(logs),
		'status':PlanStatus.PROGRESS.name
    }))

def insertRoute(item=Route):  
    # Serialize geometry and route fields to JSON if they are dict/list
    if isinstance(item.geometry, (dict, list)):
        item.geometry = json.dumps(item.geometry)
    if isinstance(item.route, (dict, list)):
        item.route = json.dumps(item.route)
    results = query_db('''
        INSERT INTO tbl_route 
        (
            "branchCode",
            "branchColor",
            "branchId",
            "branchName",
            "code", 
            "driverId",
            "driverName",
            "driverPhone",
            "driverUsername",
            "endDate",
            "endLocationAddress",
            "endLocationElevation",
            "endLocationLat",
            "endLocationLon",
            "geometry",
            "id",
            "isAllocated",
            "isDispatched",
            "isExecuting",
            "isFinish",
            "isFinishWithTrouble",
            "isFrozen",
            "maxCapacity",
            "maxWeight",
            "note",
            "planCode",
            "planId",
            "planMode",
            "planTraffic",
            "route",
            "startDate",
            "startLocationAddress",
            "startLocationElevation",
            "startLocationLat",
            "startLocationLon",
            "taskCode",
            "taskDate",
            "taskId",
            "totalDistance",
            "totalDuration",
            "totalDurationAvailable",
            "totalIdle",
            "totalOrder",
            "totalQty",
            "totalTransaction",
            "totalVolume",
            "totalWeight",
            "tourEndDate",
            "tourEndDistance",
            "tourEndDuration",
            "tourStartDate",
            "tourStartDistance",
            "tourStartDuration",
            "transporterCode",
            "transporterId",
            "transporterName",
            "transporterPhone",
            "travelMode",
            "vehicleCode",
            "vehicleId",
            "vehicleModelCode",
            "vehicleModelId",
            "vehiclePlateNumber",
            "isManualChange",
            "avoidTolls",
            "avoidFerries",
            "avoidHighways"  
        ) 
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (id) DO UPDATE 
        SET "tourEndDate" = EXCLUDED."tourEndDate", 
            "tourEndDuration" = EXCLUDED."tourEndDuration",
            "tourEndDistance" = EXCLUDED."tourEndDistance";
    ''',(
        item.branchCode,
        item.branchColor,
        item.branchId,
        item.branchName,
        item.code,
        item.driverId,
        item.driverName,
        item.driverPhone,
        item.driverUsername,
        item.endDate,
        item.endLocationAddress,
        item.endLocationElevation,
        item.endLocationLat,
        item.endLocationLon,
        item.geometry,
        item.id,
        item.isAllocated,
        item.isDispatched,
        item.isExecuting,
        item.isFinish,
        item.isFinishWithTrouble,
        item.isFrozen,
        item.maxCapacity,
        item.maxWeight,
        item.note,
        item.planCode,
        item.planId,
        item.planMode,
        item.planTraffic,
        item.route,
        item.startDate,
        item.startLocationAddress,
        item.startLocationElevation,
        item.startLocationLat,
        item.startLocationLon,
        item.taskCode,
        item.taskDate,
        item.taskId,
        item.totalDistance,
        item.totalDuration,
        item.totalDurationAvailable,
        item.totalIdle,
        item.totalOrder,
        item.totalQty,
        item.totalTransaction,
        item.totalVolume,
        item.totalWeight,
        item.tourEndDate,
        item.tourEndDistance,
        item.tourEndDuration,
        item.tourStartDate,
        item.tourStartDistance,
        item.tourStartDuration,
        item.transporterCode,
        item.transporterId,
        item.transporterName,
        item.transporterPhone,
        item.travelMode,
        item.vehicleCode,
        item.vehicleId,
        item.vehicleModelCode,
        item.vehicleModelId,
        item.vehiclePlateNumber,
        item.isManualChange,
        item.avoidTolls,
        item.avoidFerries,
        item.avoidHighways
    ), onlyExecute=True) 
    return results
    
def insertRouteOrder(item=RouteOrder):  
    """Insert or update route order with proper parameter handling"""
    
    # Helper functions for safe type conversion
    def safe_int(val, default=0):
        try:
            return int(float(val)) if val is not None else default
        except (ValueError, TypeError):
            return default
            
    def safe_float(val, default=0.0):
        try:
            return float(val) if val is not None else default
        except (ValueError, TypeError):
            return default

    # Ensure all required fields exist with defaults
    required_fields = {
        'branchCode': '', 'branchColor': '', 'branchId': '', 'branchName': '',
        'distance': 0.0, 'duration': 0.0, 'durationInTraffic': 0.0,
        'durationTotal': 0.0, 'durationUsed': 0.0, 'elevation': 0.0,
        'eta': None, 'etd': None, 'geometry': None, 'id': '',
        'idle': 0, 'isAllocated': False, 'isDispatched': False,
        'isExecuting': False, 'isFinish': False, 'isFinishWithTrouble': False,
        'isFrozen': False, 'isManualChange': False,
        'kecId': '', 'kecName': '', 'kelId': '', 'kelName': '',
        'kotaId': '', 'kotaName': '', 'lat': 0.0, 'lon': 0.0,
        'note': '', 'orderCode': '', 'orderDate': None, 'orderId': '',
        'planCode': '', 'planId': '', 'priority': 0,
        'provId': '', 'provName': '', 'qty': 0,
        'routeId': '', 'routes': None, 'skills': '',
        'sorting': 0, 'storeAddress': '', 'storeCode': '',
        'storeCustomerName': '', 'storeCustomerPhone': '',
        'storeSkills':'',
        'storeId': '', 'storeName': '',
        'taskCode': '', 'taskDate': None, 'taskId': '',
        'transaction': 0.0, 'type': '',
        'volume': 0.0, 'weight': 0.0,
        'routeCode': '', 'realSorting': 0
    }

    # Create params dict with proper defaults
    params = {}
    for field, default in required_fields.items():
        value = getattr(item, field, default)
        
        # Handle JSON serialization for geometry and routes
        if field in ['geometry', 'routes'] and isinstance(value, (dict, list)):
            value = json.dumps(value)
            
        # Handle numeric conversions
        if field in ['distance', 'duration', 'durationInTraffic', 'durationTotal', 
                    'durationUsed', 'elevation', 'lat', 'lon', 'transaction', 
                    'volume', 'weight']:
            value = safe_float(value)
        elif field in ['idle', 'priority', 'qty', 'sorting', 'realSorting']:
            value = safe_int(value)
            
        params[field] = value

    # Build SQL query with proper parameter handling
    sql = '''
        INSERT INTO tbl_route_order (
            "branchCode", "branchColor", "branchId", "branchName",
            "distance", "duration", "durationInTraffic", "durationTotal",
            "durationUsed", "elevation", "eta", "etd", "geometry", "id",
            "idle", "isAllocated", "isDispatched", "isExecuting",
            "isFinish", "isFinishWithTrouble", "isFrozen", "isManualChange",
            "kecId", "kecName", "kelId", "kelName", "kotaId", "kotaName",
            "lat", "lon", "note", "orderCode", "orderDate", "orderId",
            "planCode", "planId", "priority", "provId", "provName", "qty",
            "routeId", "routes", "skills", "sorting", "storeAddress",
            "storeCode", "storeCustomerName", "storeCustomerPhone",
            "storeSkills",
            "storeId", "storeName", "taskCode", "taskDate", "taskId",
            "transaction", "type", "volume", "weight", "routeCode", "realSorting"
        ) VALUES (
            %(branchCode)s, %(branchColor)s, %(branchId)s, %(branchName)s,
            %(distance)s, %(duration)s, %(durationInTraffic)s, %(durationTotal)s,
            %(durationUsed)s, %(elevation)s, %(eta)s, %(etd)s, %(geometry)s, %(id)s,
            %(idle)s, %(isAllocated)s, %(isDispatched)s, %(isExecuting)s,
            %(isFinish)s, %(isFinishWithTrouble)s, %(isFrozen)s, %(isManualChange)s,
            %(kecId)s, %(kecName)s, %(kelId)s, %(kelName)s, %(kotaId)s, %(kotaName)s,
            %(lat)s, %(lon)s, %(note)s, %(orderCode)s, %(orderDate)s, %(orderId)s,
            %(planCode)s, %(planId)s, %(priority)s, %(provId)s, %(provName)s, %(qty)s,
            %(routeId)s, %(routes)s, %(skills)s, %(sorting)s, %(storeAddress)s,
            %(storeCode)s, %(storeCustomerName)s, %(storeCustomerPhone)s,
            %(storeSkills)s,
            %(storeId)s, %(storeName)s, %(taskCode)s, %(taskDate)s, %(taskId)s,
            %(transaction)s, %(type)s, %(volume)s, %(weight)s, %(routeCode)s, %(realSorting)s
        )
    '''

    try:
        results = query_db(sql, params, onlyExecute=True)
        return results
    except Exception as e:
        print(f"Error inserting route order: {e}")
        print(f"Failed params: {params}")
        raise

def savePlan(item:Plan):  
    results = query_db('''
        UPDATE tbl_plan
        SET 
            "totalAllocated"=%s,
            "totalUnallocated"=%s,
            "totalDistance"=%s,
            "totalDuration"=%s,
            "totalDurationAvailable"=%s,
            "totalIdle"=%s,
            "totalQty"=%s,
            "totalStore"=%s,
            "totalTransaction"=%s,
            "totalVehicle"=%s,
            "totalVehicleUsed"=%s,
            "totalVolume"=%s,
            "totalWeight"=%s,
            "cluster"=%s,
            "isManualChange"=%s
        WHERE
            id=%s
    ''',(item.totalAllocated,item.totalUnallocated,item.totalDistance,item.totalDuration,item.totalDurationAvailable,item.totalIdle,item.totalQty,item.totalStore,item.totalTransaction,item.totalVehicle,item.totalVehicleUsed,item.totalVolume,item.totalWeight,item.cluster,item.isManualChange,item.id), onlyExecute=True) 
    return results
 