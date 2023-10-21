use std::collections::HashMap;
use std::fs::File;
use csv;

fn main() {

    let mut stop_times = csv::Reader::from_path("../gtfs_static/la_bus/stop_times.txt").expect("CSV read error");
    let mut indexed_stop_times = Box::new(HashMap::with_capacity(50000));
    let fobj = File::create("./la_stop_times_indexed.txt").expect("Failed to create a new index file");

    println!("INDEXING STOP ARRIVAL TIMES...");
    for record in stop_times.records() {
        if let Ok(record) = record {
            let key = String::from(&record[0]);
            if !indexed_stop_times.contains_key(&key) {
                indexed_stop_times.insert(key, vec![(String::from(&record[3]), String::from(&record[1]))]);
            } else {
                indexed_stop_times.get_mut(&key).unwrap().push((String::from(&record[3]), String::from(&record[1])));
            }
        }
    }
    println!("DONE");

    serde_json::to_writer(fobj, &indexed_stop_times).expect("Failed to save the indexed file");
}
