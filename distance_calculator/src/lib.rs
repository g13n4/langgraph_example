use pyo3::prelude::*;

#[pymodule]
mod distance_calculator {
    use time::Duration;

    use pyo3::prelude::*;

    use travelling_salesman::Tour;
    use travelling_salesman::get_distance_matrix;
    use travelling_salesman::simulated_annealing;

    type Location = (f64, f64);
    type TotalTravelDistance = f64;
    type NextTravelDestinationDistance = Option<f64>;
    type OutputVec = Vec<(Location, NextTravelDestinationDistance)>;

    #[pyfunction]
    fn calculate_route(
        destinations: Vec<Location>,
        destination_names: Vec<String>,
        time_to_process: Option<f32>,
    ) -> PyResult<(TotalTravelDistance, OutputVec, Vec<String>)> {
        let number_of_destinations: usize = destinations.len();

        let Tour { distance, route } = simulated_annealing::solve(
            &destinations,
            Duration::seconds_f32(time_to_process.unwrap_or(0.2)),
        );

        // We setting initial location to zeroes to create a vector of the size we need
        let mut routes_in_optimal_order: Vec<Location> = vec![(0.0, 0.0); number_of_destinations];
        let mut routes_names_in_optimal_order: Vec<String> = vec![String::from(""); number_of_destinations];
        for (route_idx, route_position) in route.iter().enumerate() {
            // We skip the very last index because TSP makes retuns are a linked list of nodes
            if route_idx == number_of_destinations { break }

            routes_in_optimal_order[*route_position] = destinations[route_idx];
            routes_names_in_optimal_order[*route_position] = destination_names[route_idx].clone();
        }

        let optimal_route_distances = get_distance_matrix(&routes_in_optimal_order);
        let mut output: OutputVec = vec![];

        for (route_idx, route) in routes_in_optimal_order.iter().enumerate() {
            let mut next_destination_idx = route_idx + 1;
            let mut next_destination_distance = None;
            if let Some(target_vector) = optimal_route_distances.get(route_idx) {
                if next_destination_idx == number_of_destinations {
                    // if it's the last node we circle it back to the first one
                    next_destination_idx = 0;
                }
                next_destination_distance = target_vector.get(next_destination_idx).copied();
            };

            output.push((*route, next_destination_distance));
        }

        Ok((distance, output, routes_names_in_optimal_order))
    }
}
