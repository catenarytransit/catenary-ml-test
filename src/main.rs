use std::collections::HashMap;
use reqwest;
use gtfs_rt;
use prost::Message;
use csv;
use chrono::prelude::*;
use csv::Position;
use serde;
use std::fs::File;

mod tz_ofs {
    pub const PDT: isize = -7 * 3600;
}

#[derive(serde::Serialize)]
struct TrainingRowData {
    vehicle_id: Option<f32>,
    is_weekend: Option<f32>,
    latitude: Option<f32>,
    longitude: Option<f32>,
    bearing: Option<f32>,
    speed: Option<f32>,
    current_time: Option<f32>,
    stop_lat: Option<f32>,
    stop_lon: Option<f32>,
    arrival_time: Option<f32>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ALL AGENCY ACCESS-POINT OVERVIEW: https://kactus.catenarymaps.org/gtfsrttimes

    // reqwest::Client::new()

    let mut writer = csv::Writer::from_path("training/training.csv")?;
    let mut stop_locs = csv::Reader::from_path("gtfs_static/la_bus/stops.txt")?;
    // let mut stop_times = csv::Reader::from_path("gtfs_static/la_bus/stop_times.txt")?;
    let stop_times_file = File::open("raw_static_data_mapper/la_stop_times_indexed.txt")?;
    let stop_times: HashMap<String, Vec<(String, String)>> = serde_json::from_reader(stop_times_file)?;

    let body = reqwest::get("https://kactus.catenarymaps.org/gtfsrt/?feed=f-metro~losangeles~bus~rt&category=vehicles");
    let bytes = body.await?.bytes().await?.to_vec();
    let content = gtfs_rt::FeedMessage::decode(bytes.as_slice())?;

    // println!("{:?}", content);
    for ent in content.entity {
        if let Some(ref vehicle) = ent.vehicle {
            let mut arrival_seconds = None;
            if let Some(timestamp) = vehicle.timestamp {
                let unix_date = DateTime::from_timestamp(timestamp as i64, 0).unwrap();
                let tz_date = unix_date.with_timezone(&FixedOffset::east_opt(tz_ofs::PDT as i32).unwrap());
                arrival_seconds = Some(tz_date.num_seconds_from_midnight() as f32);
            }

            if let Some(ref vehicle_descrp) = vehicle.vehicle {
                if let Some(ref position) = vehicle.position {
                    let mut stop_lat = None;
                    let mut stop_lon = None;
                    for record in stop_locs.records() {
                        if let Ok(record) = &record {
                            if let Some(stop_id) = &vehicle.stop_id {
                                if &record[0] == stop_id.as_str() {
                                    stop_lat = Some(record[4].replace(" ", "").parse::<f32>().unwrap());
                                    stop_lon = Some(record[5].replace(" ", "").parse::<f32>().unwrap());
                                    stop_locs.seek(Position::new())?;
                                    break;
                                }
                            }
                        } else {
                            println!("Reading CSV file record error [stops.txt]");
                        }
                    }
                    stop_locs.seek(Position::new())?;  // VERY IMPORTANT

                    if let Some(ref trip) = vehicle.trip {
                        let mut arrival_time = None;

                        if let Some(stop_id) = &vehicle.stop_id {
                            let trip_id = trip.trip_id.as_ref().expect("No trip id");
                            for (stop, arrival_time_dat) in stop_times.get(trip_id).expect(&*format!("MISSING TRIP ID {:?}", trip_id)) {
                                if stop == stop_id && !arrival_time_dat.is_empty() {
                                    let record_fmt = if arrival_time_dat.len() == 7 {
                                        format!("0{}", arrival_time_dat)
                                    } else { arrival_time_dat.clone() };
                                    arrival_time = Some(NaiveTime::parse_from_str(
                                        &*record_fmt, "%H:%M:%S"
                                    ).unwrap().num_seconds_from_midnight() as f32);
                                }
                            }

                            // for record in stop_times.records() {
                            //     if let Ok(record) = &record {
                            //         if &record[0] == trip_id.as_str() && &record[3] == stop_id.as_str() && record[1].len() != 0 {
                            //             let record_fmt = if record[1].len() == 7 {
                            //                 format!("0{}", &record[1])
                            //             } else {
                            //                 (&record[1]).to_string()
                            //             };
                            //             arrival_time = Some(NaiveTime::parse_from_str(
                            //                 &*record_fmt, "%H:%M:%S"
                            //             ).unwrap().num_seconds_from_midnight() as f32);
                            //             stop_times.seek(Position::new())?;
                            //             break;
                            //         }
                            //     } else {
                            //         println!("Reading CSV file record error [stop_times.txt]");
                            //     }
                            // }
                            // stop_times.seek(Position::new())?;
                        }

                        let weekday = NaiveDate::parse_from_str(&trip.start_date.clone().unwrap(), "%Y%m%d")
                            .unwrap().weekday();
                        // println!("{:?} {:?}", trip.trip_id, vehicle.stop_id);
                        writer.serialize(TrainingRowData {
                            vehicle_id: Some(vehicle_descrp.id.as_ref().unwrap().parse::<f32>().unwrap()),
                            is_weekend: Some(if weekday == Weekday::Sat || weekday == Weekday::Sun {1.0} else {0.0}),
                            latitude: Some(position.latitude),
                            longitude: Some(position.longitude),
                            bearing: position.bearing,
                            speed: position.speed,
                            current_time: arrival_seconds,
                            stop_lat,
                            stop_lon,
                            arrival_time,
                        }).expect("Failed to serialize");
                        writer.flush().expect("Failed to flush");
                    } else {
                        writer.serialize(TrainingRowData {
                            vehicle_id: Some(vehicle_descrp.id.as_ref().unwrap().parse::<f32>().unwrap()),
                            is_weekend: None,
                            latitude: Some(position.latitude),
                            longitude: Some(position.longitude),
                            bearing: position.bearing,
                            speed: position.speed,
                            current_time: arrival_seconds,
                            stop_lat,
                            stop_lon,
                            arrival_time: None,
                        }).expect("Failed to serialize (no trip)");
                        writer.flush().expect("Failed to flush (no trip)");
                    }
                }
            }
        }
    }

    Ok(())
}
