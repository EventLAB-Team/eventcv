fn main() {
    // A failed install (another handler already set) only costs the clean shutdown of `record`.
    let _ = ctrlc::set_handler(eventcv_cli::on_interrupt);
    std::process::exit(eventcv_cli::run(std::env::args_os()));
}
