use redis::Commands;
use qstring::QString;

pub async fn gtfsrt(qs: &str) -> Option<Vec<u8>> {
    let redisclient = redis::Client::open("redis://127.0.0.1:6379/").unwrap();
    let mut con = redisclient.get_connection().unwrap();

    let qs = QString::from(qs); // "?feed=ferret"
    let feed = qs.get("feed");

    match feed {
        Some(feed) => {
            let category = qs.get("category");

            //HttpResponse::Ok().body(format!("Requested {}/{}", feed, category))

            match category {
                Some(category) => {
                    let doesexist =
                        con.get::<String, u64>(format!("gtfsrttime|{}|{}", &feed, &category));

                    match doesexist {
                        Ok(_timeofcache) => {
                            let data = con
                                .get::<String, Vec<u8>>(format!("gtfsrt|{}|{}", &feed, &category));

                            match data {
                                Ok(data) => {
                                    println!("DATA!");

                                    return Some(data);
                                }
                                Err(e) => {
                                    println!("Error: {:?}", e);
                                }
                            }
                        }
                        Err(_e) => {
                            println!("Error in connecting to redis");
                        }
                    }
                }
                None => {
                    println!("Error: No category specified");
                }
            }
        }
        None => {
            println!("Error: No feed specified");
        }
    }

    return None;
}
