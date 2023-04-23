#![allow(warnings, unused)]
#![feature(once_cell)]
#![feature(thread_local)]
#![feature(new_uninit)]

macro_rules! simple_empty_finalize_trace {
    ($($T:ty),*) => {
        $(
            impl Finalize for $T {}
            unsafe impl Trace for $T { unsafe_empty_trace!(); }
        )*
    }
}

mod ext {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

// pub mod calcium {
    use std::fmt::{Debug, Display};

    use auto_ops::impl_op_ex;
    use std::cell::LazyCell;
    use std::ffi::{c_long, c_ulong, c_void, CStr, CString};

    use ext::{
        ca_add, ca_check_equal, ca_clear, ca_ctx_clear, ca_ctx_init, ca_ctx_struct, ca_div,
        ca_get_fmpz, ca_get_str, ca_init, ca_mul, ca_one, ca_pi, ca_set_d, ca_set_fmpq, ca_set_si,
        ca_set_ui, ca_struct, ca_sub, ca_zero, flint_cleanup, flint_free, fmpq, fmpq_clear,
        fmpq_init, fmpq_set_str, fmpz, fmpz_abs_fits_ui, fmpz_clear, fmpz_get_ui, fmpz_init,
        fmpz_sgn, truth_t_T_FALSE, truth_t_T_TRUE,
    };
    use gc::{unsafe_empty_trace, Finalize, Trace};

    pub struct Context {
        ctx: *mut ca_ctx_struct,
    }
    impl Drop for Context {
        fn drop(&mut self) {
            unsafe {
                ca_ctx_clear(self.ctx);
                flint_cleanup();
                drop(Box::from_raw(self.ctx));
            }
        }
    }

    #[thread_local]
    pub static mut CALCIUM_CTX: LazyCell<Context> = LazyCell::new(|| unsafe {
        let mut ctx = Box::<ca_ctx_struct>::new_uninit();
        ca_ctx_init(ctx.as_mut_ptr());
        Context {
            ctx: Box::into_raw(ctx.assume_init()),
        }
    });

    pub struct Number {
        data: *mut ca_struct,
    }

    simple_empty_finalize_trace![Number];

    // TODO: mutable update like +=?
    impl Number {
        pub fn new() -> Number {
            unsafe {
                let mut x = Box::<ca_struct>::new_uninit();
                ca_init(x.as_mut_ptr(), CALCIUM_CTX.ctx);
                Number {
                    data: Box::into_raw(x.assume_init()),
                }
            }
        }
        pub fn zero() -> Number {
            let x = Number::new();
            unsafe {
                ca_zero(x.data, CALCIUM_CTX.ctx);
            }
            x
        }
        pub fn one() -> Number {
            let x = Number::new();
            unsafe {
                ca_one(x.data, CALCIUM_CTX.ctx);
            }
            x
        }
        pub fn pi() -> Number {
            let x = Number::new();
            unsafe {
                ca_pi(x.data, CALCIUM_CTX.ctx);
            }
            x
        }
        pub fn from_clong(i: c_long) -> Number {
            let x = Number::new();
            unsafe {
                ca_set_si(x.data, i, CALCIUM_CTX.ctx);
            }
            x
        }
        pub fn from_ulong(u: c_ulong) -> Number {
            let x = Number::new();
            unsafe {
                ca_set_ui(x.data, u, CALCIUM_CTX.ctx);
            }
            x
        }
        pub fn from_f64(f: f64) -> Number {
            let x = Number::new();
            unsafe {
                ca_set_d(x.data, f, CALCIUM_CTX.ctx);
            }
            x
        }
        pub fn from_str(s: &str) -> Option<Number> {
            unsafe {
                let mut x = Box::<fmpq>::new_uninit();
                fmpq_init(x.as_mut_ptr());
                let x = Box::into_raw(x.assume_init());
                let cs = CString::new(s).unwrap();
                let e = fmpq_set_str(x, cs.as_ptr(), 10);
                let n = Number::new();
                ca_set_fmpq(n.data, x, CALCIUM_CTX.ctx);
                fmpq_clear(x);
                drop(Box::from_raw(x));
                if e == 0 {
                    Some(n)
                } else {
                    None // maybe use an error message in the future
                }
            }
        }
    }
    impl_op_ex!(+ |a: &Number, b: &Number| -> Number {
        let res = Number::new();
        unsafe { ca_add(res.data, a.data, b.data, CALCIUM_CTX.ctx); }
        res
    });
    impl_op_ex!(-|a: &Number, b: &Number| -> Number {
        let res = Number::new();
        unsafe {
            ca_sub(res.data, a.data, b.data, CALCIUM_CTX.ctx);
        }
        res
    });
    impl_op_ex!(*|a: &Number, b: &Number| -> Number {
        let res = Number::new();
        unsafe {
            ca_mul(res.data, a.data, b.data, CALCIUM_CTX.ctx);
        }
        res
    });
    impl_op_ex!(/ |a: &Number, b: &Number| -> Number {
        let res = Number::new();
        unsafe { ca_div(res.data, a.data, b.data, CALCIUM_CTX.ctx); }
        res
    });
    impl Into<usize> for Number {
        fn into(self) -> usize {
            (&self).into()
        }
    }
    impl Into<usize> for &Number {
        fn into(self) -> usize {
            unsafe {
                let mut x = Box::<fmpz>::new_uninit();
                fmpz_init(x.as_mut_ptr());
                let x = Box::into_raw(x.assume_init());
                let e = ca_get_fmpz(x, self.data, CALCIUM_CTX.ctx);
                if e == 0 {
                    fmpz_clear(x);
                    drop(Box::from_raw(x));
                    panic!("is not integer");
                }
                if fmpz_sgn(x) < 0 || fmpz_abs_fits_ui(x) == 0 {
                    panic!("does not fit into usize");
                }
                let ui = fmpz_get_ui(x);
                fmpz_clear(x);
                drop(Box::from_raw(x));
                ui as usize
            }
        }
    }
    impl Drop for Number {
        fn drop(&mut self) {
            unsafe {
                ca_clear(self.data, CALCIUM_CTX.ctx);
                drop(Box::from_raw(self.data));
            }
        }
    }
    impl Eq for Number {}
    impl PartialEq for Number {
        fn eq(&self, other: &Self) -> bool {
            unsafe {
                let truth = ca_check_equal(self.data, other.data, CALCIUM_CTX.ctx);
                match truth {
                    truth_t_T_TRUE => true,
                    truth_t_T_FALSE => false,
                    _ => panic!("incomparable"),
                }
            }
        }
    }
    impl Display for Number {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            unsafe {
                let s = ca_get_str(self.data, CALCIUM_CTX.ctx);
                let res = f.write_str(CStr::from_ptr(s).to_str().unwrap());
                flint_free(s as *mut c_void);
                res
            }
        }
    }
    impl Debug for Number {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            Display::fmt(&self, f)
        }
    }
// }