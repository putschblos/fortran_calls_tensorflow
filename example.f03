MODULE example
    USE ISO_C_BINDING
    IMPLICIT NONE

    !========================================================================================================================
    !> Call the Model from the C++ shared library.
    !========================================================================================================================
    INTERFACE
        SUBROUTINE modelres(xptr,yptr,sizex,sizey) BIND(C, name='run')
            IMPORT ::c_ptr
            IMPORT ::C_INT
            TYPE(C_ptr), VALUE :: xptr
            TYPE(C_ptr), VALUE :: yptr
            INTEGER(C_INT), VALUE ::  sizey
            INTEGER(C_INT), VALUE ::  sizex
        END SUBROUTINE
    END INTERFACE
    !------------------------------------------------------------------------------------------------------------------------
CONTAINS

    FUNCTION localWrapper(modelInput) RESULT(modelOutput)

        !------------------------------------------------------------------------------------------------------------------------
        ! INPUT/OUTPUT Variables
        REAL(kind=RP), DIMENSION(1:15),INTENT(IN) ::  modelInput
        REAL(kind=RP)                             ::  modelOutput
        !------------------------------------------------------------------------------------------------------------------------
        ! LOCAL VARIABLES
        REAL(C_double),TARGET                     ::  x(15)
        REAL(C_double),TARGET                     ::  y(1)
        INTEGER                                   ::  x_size,y_size
        Type(C_ptr)                               ::  xptr,yptr
        !------------------------------------------------------------------------------------------------------------------------
        x_size = 15
        y_size = 1


        x = modelInput

        yptr = c_loc(y(1))
        xptr = c_loc(x(1))
        CALL modelres(xptr,yptr,x_size,y_size)

        modelOutput=y(1)
    END FUNCTION localWrapper

END MODULE example
