mod external;

use std::collections::HashMap;
use reqwest;
use gtfs_rt;
use prost::Message;
use csv;
use chrono::prelude::*;
use csv::{Position, Reader};
use std::fs::File;
use std::time::Duration;
use polars::prelude::*;


pub const IS_SERVER: bool = false;  // true: use local memory Redis cache; false: use HTTP req

mod tz_ofs {
    pub const PDT: isize = -7 * 3600;
}

// #[derive(serde::Serialize)]
// struct TrainingRowDataOld {
//     vehicle_id: Option<f32>,
//     is_weekend: Option<f32>,
//     latitude: Option<f32>,
//     longitude: Option<f32>,
//     bearing: Option<f32>,
//     speed: Option<f32>,
//     current_time: Option<f32>,
//     stop_lat: Option<f32>,
//     stop_lon: Option<f32>,
//     arrival_time: Option<f32>,
//
//     actual_arrival_time: Option<i32>,  // TODO: currently just represents the current status of the vehicle
// }
//
// struct TrainingRowData2 {
//     vehicle_id: Series,
//     is_weekend: Series,
//     latitude: Series,
//     longitude: Series,
//     bearing: Series,
//     speed: Series,
//     current_time: Series,
//     stop_lat: Series,
//     stop_lon: Series,
//     arrival_time: Series,
//
//     actual_arrival_time: Series,  // TODO: currently just represents the current status of the vehicle
// }
//
// impl TrainingRowData2 {
//     pub fn clear(&self) {
//         self.vehicle_id.clear();
//         self.is_weekend.clear();
//         self.latitude.clear();
//         self.longitude.clear();
//         self.bearing.clear();
//         self.speed.clear();
//         self.current_time.clear();
//         self.stop_lat.clear();
//         self.stop_lon.clear();
//         self.arrival_time.clear();
//
//         self.actual_arrival_time.clear();
//     }
//
//     pub fn to_dataframe(&self) -> DataFrame {
//         DataFrame::new(vec![
//             self.vehicle_id,
//             self.is_weekend,
//             self.latitude,
//             self.longitude,
//             self.bearing,
//             self.speed,
//             self.current_time,
//             self.stop_lat,
//             self.stop_lon,
//             self.arrival_time,
//             self.actual_arrival_time,
//         ])?
//     }
// }

#[derive(Default)]
struct TrainingRowData {
    vehicle_id: Vec<f32>,
    is_weekend: Vec<f32>,
    latitude: Vec<f32>,
    longitude: Vec<f32>,
    bearing: Vec<Option<f32>>,
    speed: Vec<f32>,
    current_time: Vec<f32>,
    stop_lat: Vec<f32>,
    stop_lon: Vec<f32>,
    arrival_time: Vec<f32>,
    actual_arrival_time: Vec<Option<f32>>,  // TODO: currently just represents the current status of the vehicle
}

impl TrainingRowData {
    fn to_dataframe(&self) -> PolarsResult<DataFrame> {
        DataFrame::new(vec![
            Series::new("vehicle_id", &self.vehicle_id),
            Series::new("is_weekend", &self.is_weekend),
            Series::new("latitude", &self.latitude),
            Series::new("longitude", &self.longitude),
            Series::new("bearing", &self.bearing),
            Series::new("speed", &self.speed),
            Series::new("current_time", &self.current_time),
            Series::new("stop_lat", &self.stop_lat),
            Series::new("stop_lon", &self.stop_lon),
            Series::new("arrival_time", &self.arrival_time),
            Series::new("actual_arrival_time", &self.actual_arrival_time),
        ])
    }

    fn clear(&mut self) {
        self.vehicle_id.clear();
        self.is_weekend.clear();
        self.latitude.clear();
        self.longitude.clear();
        self.bearing.clear();
        self.speed.clear();
        self.current_time.clear();
        self.stop_lat.clear();
        self.stop_lon.clear();
        self.arrival_time.clear();
        self.actual_arrival_time.clear();
    }

    fn dataframe_empty() -> PolarsResult<DataFrame> {
        DataFrame::new(vec![
            Series::new_empty("vehicle_id", &DataType::Float32),
            Series::new_empty("is_weekend", &DataType::Float32),
            Series::new_empty("latitude", &DataType::Float32),
            Series::new_empty("longitude", &DataType::Float32),
            Series::new_empty("bearing", &DataType::Float32),
            Series::new_empty("speed", &DataType::Float32),
            Series::new_empty("current_time", &DataType::Float32),
            Series::new_empty("stop_lat", &DataType::Float32),
            Series::new_empty("stop_lon", &DataType::Float32),
            Series::new_empty("arrival_time", &DataType::Float32),
            Series::new_empty("actual_arrival_time", &DataType::Float32),
        ])
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("INITIALIZING...");

    // reqwest::Client::new()

    // let mut writer = csv::Writer::from_path("training/training.csv")?;

    // let mut file = File::create("training/testing.csv")?;
    // let mut training_csv = CsvWriter::new(file)
    //     .has_header(true)
    //     .with_delimiter(b',');
    //
    // let mut df_headers = DataFrame::new(vec![
    //     Series::new("Fruit", &["A"]),
    //     Series::new("Color", &["B"]),
    // ])?;
    // training_csv.finish(&mut df_headers);
    // training_csv = training_csv.has_header(false);
    //
    // let mut df = df!(
    //     "Fruit" => &["Apple", "Apple", "Pear"],
    //     "Color" => &["Red", "Yellow", "Green"]
    // )?;
    //
    // training_csv.finish(&mut df)?;
    // println!("DF2 {:?}", df);

    let mut main_df = TrainingRowData::dataframe_empty()?;
    let mut trd = TrainingRowData::default();

    let mut stop_locs = csv::Reader::from_path("stops.txt")?;
    let stop_times_file = File::open("raw_static_data_mapper/la_stop_times_indexed.txt")?;
    let stop_times: HashMap<String, Vec<(String, String)>> = serde_json::from_reader(stop_times_file)?;

    let mut interval = tokio::time::interval(Duration::new(60, 0));
    let mut min = 0;
    loop {
        println!("MINUTE INTERVAL {:?}", min);
        min += 1;
        interval.tick().await;

        println!("SNAPSHOT BEGIN");
        let bytes = if IS_SERVER {
            external::gtfsrt("?feed=f-metro~losangeles~bus~rt&category=vehicles").await.expect("ERROR: CANNOT ACCESS REDIS CACHED RT DATA")
        } else {
            let body = reqwest::get("https://kactus.catenarymaps.org/gtfsrt/?feed=f-metro~losangeles~bus~rt&category=vehicles");
            body.await?.bytes().await?.to_vec()
        };
        let content = gtfs_rt::FeedMessage::decode(bytes.as_slice())?;

        main_df = snapshot(&mut trd, main_df, &mut stop_locs, &stop_times, &content).await?;
        main_df.extend(&mut trd.to_dataframe().unwrap())?;

        let file = File::create("../training/one_hour_data.csv")?;
        let mut training_csv = CsvWriter::new(&file)
            .has_header(true)
            .with_delimiter(b',');
        training_csv.finish(&mut main_df)?;
        trd.clear();
        println!("SNAPSHOT END");
    }
}

async fn snapshot(
    trd: &mut TrainingRowData, mut main_df: DataFrame,
    stop_locs: &mut Reader<File>, stop_times: &HashMap<String, Vec<(String, String)>>, content: &gtfs_rt::FeedMessage
) -> Result<DataFrame, Box<dyn std::error::Error>> {
    // ALL AGENCY ACCESS-POINT OVERVIEW: https://kactus.catenarymaps.org/gtfsrttimes

    // println!("{:?}", content);
    for ent in &content.entity {
        if let Some(ref vehicle) = ent.vehicle {
            if let Some(ref vehicle_descrp) = vehicle.vehicle {
                if let Some(ref position) = vehicle.position {
                    let mut timestamp_seconds = None;
                    if let Some(timestamp) = vehicle.timestamp {
                        let unix_date = DateTime::from_timestamp(timestamp as i64, 0).unwrap();
                        let tz_date = unix_date.with_timezone(&FixedOffset::east_opt(tz_ofs::PDT as i32).unwrap());
                        timestamp_seconds = Some(tz_date.num_seconds_from_midnight() as f32);
                    }

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

                                    let hours = (&record_fmt[0..=1]).parse::<u32>().unwrap();
                                    let minutes = (&record_fmt[3..=4]).parse::<u32>().unwrap();
                                    let seconds = (&record_fmt[6..=7]).parse::<u32>().unwrap();
                                    // println!("{}:{}:{} = {}", hours, minutes, seconds, hours*3600+minutes*60+seconds);

                                    arrival_time = Some((hours*3600+minutes*60+seconds) as f32);
                                    // arrival_time = Some(NaiveTime::parse_from_str(
                                    //     &*record_fmt, "%H:%M:%S"
                                    // ).unwrap().num_seconds_from_midnight() as f32);
                                }
                            }
                        }

                        let weekday = NaiveDate::parse_from_str(&trip.start_date.clone().unwrap(), "%Y%m%d")
                            .unwrap().weekday();
                        // println!("{:?} {:?}", trip.trip_id, vehicle.stop_id);

                        if let Some(1) = vehicle.current_status { // current status: stopped_at
                            // search for all empty actual arrival time for that vehicle id and fill in the current time

                            main_df = main_df
                                .lazy()
                                .with_column(
                                    when(
                                        col("vehicle_id").eq(lit(vehicle_descrp.id.as_ref().unwrap().parse::<f32>().unwrap()))
                                            .and(col("actual_arrival_time").eq(lit(Null {})))
                                    )
                                        .then(lit(timestamp_seconds.unwrap()))  // final actual arrival time propogate immediately to all previous null value of this vehID
                                        .when(col("vehicle_id").eq(lit(vehicle_descrp.id.as_ref().unwrap().parse::<f32>().unwrap())))
                                        .then(col("actual_arrival_time")+lit(1.0))  // store the previous arrival time
                                        .otherwise(col("actual_arrival_time"))  // do nothing
                                        .alias("actual_arrival_time")
                                )
                                .collect().unwrap();

                            // println!("{:?}");

                            // let mut temp_df = main_df
                            //     .filter(
                            //         &main_df
                            //             .column("vehicle_id").unwrap()
                            //             .equal(vehicle_descrp.id.as_ref().unwrap().parse::<f32>().unwrap()).unwrap()
                            //     ).unwrap();
                            // let updated_actual_arrival_time = temp_df
                            //     .apply("actual_arrival_time", |series: &Series| {
                            //         series
                            //             .f32().unwrap()
                            //             .into_iter()
                            //             .map(|val: Option<f32>| {
                            //                 if let Some(val) = val {
                            //                     Some(val+1.0)
                            //                 } else {
                            //                     Some(100.0)
                            //                 }
                            //                 // println!("{:?} {:?}", vehicle_descrp.id, val);
                            //                 // val+1
                            //                 // if let Some(val) = val { Some(val) }
                            //                 // else { Some(-666.0f32) }
                            //             })
                            //             .collect::<Float32Chunked>()
                            //             .into_series()
                            //     }).unwrap()
                            //     .column("actual_arrival_time").unwrap();

                            // println!("{:?} {:?}", vehicle_descrp.id, updated_actual_arrival_time);
                            // main_df.with_column(updated_actual_arrival_time).unwrap();

                            trd.vehicle_id.push(vehicle_descrp.id.as_ref().unwrap().parse::<f32>().unwrap());
                            trd.is_weekend.push(if weekday == Weekday::Sat || weekday == Weekday::Sun {1.0f32} else {0.0f32});
                            trd.latitude.push(position.latitude);
                            trd.longitude.push(position.longitude);
                            trd.bearing.push(position.bearing);
                            trd.speed.push(position.speed.unwrap());
                            trd.current_time.push(timestamp_seconds.unwrap());
                            trd.stop_lat.push(stop_lat.unwrap());
                            trd.stop_lon.push(stop_lon.unwrap());
                            trd.arrival_time.push(arrival_time.unwrap());
                            trd.actual_arrival_time.push(Some(timestamp_seconds.unwrap()));  // if *still* at stop: timestamp == actual arrival time
                        } else {
                            trd.vehicle_id.push(vehicle_descrp.id.as_ref().unwrap().parse::<f32>().unwrap());
                            trd.is_weekend.push(if weekday == Weekday::Sat || weekday == Weekday::Sun {1.0f32} else {0.0f32});
                            trd.latitude.push(position.latitude);
                            trd.longitude.push(position.longitude);
                            trd.bearing.push(position.bearing);
                            trd.speed.push(position.speed.unwrap());
                            trd.current_time.push(timestamp_seconds.unwrap());
                            trd.stop_lat.push(stop_lat.unwrap());
                            trd.stop_lon.push(stop_lon.unwrap());
                            trd.arrival_time.push(arrival_time.unwrap());
                            trd.actual_arrival_time.push(None);
                        }

                        // writer.serialize(TrainingRowDataOld {
                        //     vehicle_id: Some(vehicle_descrp.id.as_ref().unwrap().parse::<f32>().unwrap()),
                        //     is_weekend: Some(if weekday == Weekday::Sat || weekday == Weekday::Sun {1.0} else {0.0}),
                        //     latitude: Some(position.latitude),
                        //     longitude: Some(position.longitude),
                        //     bearing: position.bearing,
                        //     speed: position.speed,
                        //     current_time: arrival_seconds,
                        //     stop_lat,
                        //     stop_lon,
                        //     arrival_time,
                        //     actual_arrival_time: vehicle.current_status,
                        // }).expect("Failed to serialize");
                        // writer.flush().expect("Failed to flush");
                    }
                }
            }
        }
    }

    Ok(main_df)
}
