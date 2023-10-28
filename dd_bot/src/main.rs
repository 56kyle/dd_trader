use std::sync::{Arc, Mutex};
use std::str;

use enigo::*;

// import Rocket
#[macro_use]
extern crate rocket;
use rocket::State;

use rocket::futures::stream::iter;
use rocket::response::stream::EventStream;

use rocket::response::stream::Event;

mod database_functions;
mod enigo_functions;
mod trading_functions;

pub enum ReadyState {
    False,
    True,
    Starting,
}

pub struct TradeBotInfo {
    ready: ReadyState,
    id: String,
}

#[derive(Clone)]
pub struct Trader {
    in_game_id: String,
    discord_channel_id: String,
    discord_id: String,
    item_images: Vec<String>,
    info_images: Vec<String>,
    gold: i32,
    has_paid_gold_fee: bool,
}

pub enum TradersContainer {
    ActiveTraders(Vec<Trader>),
}

impl TradersContainer {
    fn append(&mut self, trader: Trader) {
        match self {
            TradersContainer::ActiveTraders(traders) => {
                traders.push(trader);
            }
        }
    }

    fn get_trader_by_in_game_id(&self, in_game_id: &str) -> Option<&Trader> {
        match self {
            TradersContainer::ActiveTraders(traders) => traders
                .iter()
                .find(|trader| trader.in_game_id == in_game_id),
        }
    }

    fn get_other_trader_in_channel(
        &self,
        discord_id: &str,
        discord_channel_id: &str,
    ) -> Option<&Trader> {
        match self {
            TradersContainer::ActiveTraders(traders) => {
                for trader in traders.iter() {
                    if trader.discord_channel_id == discord_channel_id
                        && trader.discord_id != discord_id
                    {
                        return Some(trader);
                    }
                }
            }
        }
        None
    }

    fn update_gold_fee_status(&mut self, in_game_id: &str, new_status: bool) {
        match self {
            TradersContainer::ActiveTraders(traders) => {
                if let Some(trader) = traders
                    .iter_mut()
                    .find(|trader| trader.in_game_id == in_game_id)
                {
                    trader.has_paid_gold_fee = new_status;
                }
            }
        }
    }

    fn set_in_game_id_by_discord_info(
        &mut self,
        new_in_game_id: &str,
        discord_id: &str,
        discord_channel_id: &str,
    ) {
        match self {
            TradersContainer::ActiveTraders(traders) => {
                for trader in traders.iter_mut() {
                    if trader.discord_id == discord_id
                        && trader.discord_channel_id == discord_channel_id
                    {
                        trader.in_game_id = new_in_game_id.to_string();
                        println!(
                            "Updated in_game_id for discord_id {}: {}",
                            discord_id, new_in_game_id
                        );
                        break;
                    }
                }
            }
        }
    }
}

#[get("/gold_fee/<in_game_id>/<discord_channel_id>/<discord_id>")]
fn gold_fee(
    in_game_id: &str,
    discord_channel_id: &str,
    discord_id: &str,
    enigo: &State<Arc<Mutex<Enigo>>>,
    bot_info: &State<Arc<Mutex<TradeBotInfo>>>,
    traders_container: &State<Arc<Mutex<TradersContainer>>>,
) -> EventStream![] {
    {
        let mut traders = traders_container.lock().unwrap();
        traders.set_in_game_id_by_discord_info(in_game_id, discord_id, discord_channel_id);
    }

    println!("testing");
    
    let stream = iter(vec![
        // First stream
        {
            println!("testing1");
            let info = bot_info.lock().unwrap();
            match info.ready {
                ReadyState::False => Event::data("TradeBot not ready"),
                ReadyState::Starting => Event::data("TradeBot is starting. Please wait 2 minutes."),
                ReadyState::True => Event::data("TradeBot ready"),
            }
        },
        // Second stream
        {
            println!("testing3");
            match trading_functions::collect_gold_fee(
                enigo,
                bot_info,
                in_game_id,
                traders_container,
            ) {
                Ok(message) => Event::data(message),
                Err(err) => Event::data(err),
            }
        },
    ]);

    EventStream::from(stream)
}

#[get("/deposit/<in_game_id>/<discord_channel_id>/<discord_id>")]
fn deposit(
    in_game_id: String,
    discord_channel_id: String,
    discord_id: String,
    enigo: &State<Arc<Mutex<Enigo>>>,
    bot_info: &State<Arc<Mutex<TradeBotInfo>>>,
    traders_container: &State<Arc<Mutex<TradersContainer>>>,
) -> EventStream![] {
    {
        let mut traders = traders_container.lock().unwrap();
        traders.set_in_game_id_by_discord_info(&in_game_id, &discord_id, &discord_channel_id);
    }

    let stream = iter(vec![
        // First stream
        {
            let info = bot_info.lock().unwrap();
            match info.ready {
                ReadyState::False => Event::data("TradeBot not ready"),
                ReadyState::Starting => Event::data("TradeBot is starting. Please wait 2 minutes."),
                ReadyState::True => Event::data("TradeBot ready"),
            }
        },
        {
            // Dereference `State` and clone the inner `Arc`.
            let enigo_cloned = enigo.inner().clone();
            let bot_info_cloned = bot_info.inner().clone();
            let traders_container_cloned = traders_container.inner().clone();

            let result = tokio::runtime::Runtime::new().unwrap().block_on(async {
                let inner_result = tokio::spawn(async move {
                    // Handle potential blocking/synchronous code
                    let blocking_result =
                        tokio::task::spawn_blocking(
                            move || match trading_functions::deposit(
                                (&enigo_cloned).into(),
                                (&bot_info_cloned).into(),
                                &in_game_id,
                                (&traders_container_cloned).into(),
                            ) {
                                Ok(_) => Ok(String::from("Trade successful!")),
                                Err(err) => Err(String::from(err)),
                            },
                        )
                        .await;

                    // Propagate the result of the blocking operation
                    match blocking_result {
                        Ok(Ok(s)) => Ok(s),
                        Ok(Err(e)) => Err(e),
                        Err(_) => Err("Task failed".to_string()),
                    }
                })
                .await;
                inner_result
            });
            match result {
                Ok(Ok(s)) => Event::data(s),
                Ok(Err(e)) => Event::data(e),
                Err(_) => Event::data("Task failed".to_string()),
            }
        },
    ]);
    EventStream::from(stream)
}

#[get("/claim_items/<in_game_id>/<discord_channel_id>/<discord_id>")]
fn claim_items(
    in_game_id: String,
    discord_channel_id: &str,
    discord_id: &str,
    enigo: &State<Arc<Mutex<Enigo>>>,
    bot_info: &State<Arc<Mutex<TradeBotInfo>>>,
    traders_container: &State<Arc<Mutex<TradersContainer>>>,
) -> String {
    {
        let mut traders = traders_container.lock().unwrap();
        traders.set_in_game_id_by_discord_info(in_game_id.as_str(), discord_id, discord_channel_id);
    }

    // Dereference `State` and clone the inner `Arc`.
    let enigo_cloned = enigo.inner().clone();
    let bot_info_cloned = bot_info.inner().clone();
    let traders_container_cloned = traders_container.inner().clone();
    let in_game_id_cloned = in_game_id.clone();

    // Spawning a new asynchronous task
    tokio::spawn(async move {
        // Using spawn_blocking to handle potential blocking/synchronous code
        let result = tokio::task::spawn_blocking(move || {
            match trading_functions::claim_items(
                enigo_cloned,
                bot_info_cloned,
                in_game_id_cloned.as_ref(),
                traders_container_cloned,
            ) {
                Ok(_) => return String::from("Trade successful!"),
                Err(err) => {
                    if err == String::from("No items left in escrow") {
                        return String::from("All items traded");
                    }
                    return err;
                }
            }
        })
        .await;

        tokio::task::yield_now().await;

        // Log the result or handle it further, based on requirements
        match result {
            Ok(s) => {
                println!("Trade result: {}", s);
            }
            Err(e) => eprintln!("Trade error: {:?}", e),
        }
    });
    {
        let info = bot_info.lock().unwrap();
        match info.ready {
            ReadyState::False => return String::from("TradeBot not ready"),
            ReadyState::Starting => {
                return String::from("TradeBot is starting. Please wait 2 minutes and try again.")
            }
            ReadyState::True => return String::from("TradeBot ready"),
        }
    } // Lock is released here as the MutexGuard goes out of scope
}

#[get("/claim_gold/<in_game_id>/<discord_channel_id>/<discord_id>")]
fn claim_gold(
    in_game_id: String,
    discord_channel_id: &str,
    discord_id: &str,
    enigo: &State<Arc<Mutex<Enigo>>>,
    bot_info: &State<Arc<Mutex<TradeBotInfo>>>,
    traders_container: &State<Arc<Mutex<TradersContainer>>>,
) -> String {
    {
        let mut traders = traders_container.lock().unwrap();
        traders.set_in_game_id_by_discord_info(in_game_id.as_str(), discord_id, discord_channel_id);
    }

    let enigo_cloned = enigo.inner().clone();
    let bot_info_cloned = bot_info.inner().clone();
    let traders_container_cloned = traders_container.inner().clone();
    let in_game_id_cloned = in_game_id.clone();

    // Spawning a new asynchronous task
    tokio::spawn(async move {
        // Using spawn_blocking to handle potential blocking/synchronous code
        let result = tokio::task::spawn_blocking(move || {
            match trading_functions::claim_gold(
                enigo_cloned,
                bot_info_cloned,
                in_game_id_cloned.as_ref(),
                traders_container_cloned,
            ) {
                Ok(_) => return String::from("Trade successful!"),
                Err(err) => {
                    if err == String::from("No items left in escrow") {
                        return String::from("All items traded");
                    }
                    return err;
                }
            }
        })
        .await;

        tokio::task::yield_now().await;

        // Log the result or handle it further, based on requirements
        match result {
            Ok(s) => {
                println!("Trade result: {}", s);
            }
            Err(e) => eprintln!("Trade error: {:?}", e),
        }
    });

    {
        let info = bot_info.lock().unwrap();
        match info.ready {
            ReadyState::False => return String::from("TradeBot not ready"),
            ReadyState::Starting => {
                return String::from("TradeBot is starting. Please wait 2 minutes and try again.")
            }
            ReadyState::True => return String::from("TradeBot ready"),
        }
    } // Lock is released here as the MutexGuard goes out of scope
}

fn rocket() -> rocket::Rocket<rocket::Build> {
    // Create 2 instances of enigo because Enigo does not implement Copy.
    let enigo = Arc::new(Mutex::new(Enigo::new()));

    let bot_info = Arc::new(Mutex::new(TradeBotInfo {
        ready: ReadyState::True,
        id: "".to_string(),
    }));

    let traders_container = Arc::new(Mutex::new(TradersContainer::ActiveTraders(Vec::new())));

    match database_functions::populate_traders_from_db(&traders_container) {
        Ok(_) => println!("Populated trades containter!"),
        Err(_) => println!("Could not populate traders containter."),
    }

    // Clone the Arc for use in main_func
    let bot_info_clone = bot_info.clone();

    // Spawn the open game function as a separate task
    /*
    tokio::spawn(async move {
        trading_functions::open_game_go_to_lobby(bot_info_clone).await;
    });
    */

    rocket::build()
        .manage(enigo) // Add the enigo as managed state
        .manage(bot_info) // Add the bot_info as managed state
        .manage(traders_container) // Add the traders_container as managed state
        .mount(
            "/",
            routes![gold_fee, deposit, claim_items, claim_gold],
        )
}

#[rocket::main]
async fn main() {
    // Simply launch Rocket in the main function
    let rocket_instance = rocket();
    if let Err(err) = rocket_instance.launch().await {
        eprintln!("Rocket server error: {}", err);
    }
}
